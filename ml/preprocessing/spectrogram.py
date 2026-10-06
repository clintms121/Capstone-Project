"""Whiten strain and produce fixed-size, log-power spectrograms.

Loads directly from GWOSC-format HDF5 via h5py rather than gwpy: gwpy/gwosc
pull in a compiled dependency (igwn-segments) that needs an MSVC toolchain
not available on this machine. This reimplements the standard
whiten -> bandpass -> spectrogram chain in numpy/scipy -- the same approach
used in LIGO's own Open Data Workshop tutorials, just without gwpy's
convenience wrappers (and without its true constant-Q transform; see
make_spectrogram).

Both LIGO strain segments and Gravity Spy glitches must go through this same
transform so they share input shape for ml/classifier/model.py.
"""

from __future__ import annotations

import h5py
import numpy as np
from scipy import signal
from scipy.ndimage import zoom


def load_gwosc_strain(path: str) -> tuple[np.ndarray, float, float]:
    """Returns (strain, dt, gps_start) from a GWOSC-format strain HDF5 file."""
    with h5py.File(path, "r") as f:
        dset = f["strain/Strain"]
        strain = dset[:]
        dt = float(dset.attrs["Xspacing"])
        gps_start = float(dset.attrs["Xstart"])
    return strain, dt, gps_start


def whiten(strain: np.ndarray, dt: float, nperseg: int = 4096) -> np.ndarray:
    """Whiten strain against its own Welch PSD estimate.

    In a full pipeline the PSD should come from a longer, separate
    off-source stretch; here it's estimated from the same segment being
    whitened, which is fine for a single demo clip but not for real
    training data (the event itself biases its own PSD estimate).
    """
    fs = 1.0 / dt
    freqs, psd = signal.welch(strain, fs=fs, nperseg=nperseg)

    n = len(strain)
    strain_fft = np.fft.rfft(strain)
    fft_freqs = np.fft.rfftfreq(n, d=dt)

    asd_interp = np.interp(fft_freqs, freqs, np.sqrt(psd))
    asd_interp[asd_interp == 0] = np.inf  # avoid divide-by-zero at DC

    white_fft = strain_fft / asd_interp
    white = np.fft.irfft(white_fft, n=n)
    return white / np.std(white)  # Welch normalization leaves an arbitrary scale


def bandpass(strain: np.ndarray, dt: float, low: float = 20.0, high: float = 500.0) -> np.ndarray:
    fs = 1.0 / dt
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, strain)


def make_spectrogram(
    strain: np.ndarray,
    dt: float,
    gps_start: float,
    event_gps: float,
    window_s: float = 1.0,
    freq_range: tuple[float, float] = (20.0, 500.0),
    out_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Whiten + bandpass `strain`, crop to `window_s` around `event_gps`, and
    return a (freq, time) spectrogram resampled to a fixed `out_size` grid,
    log-power, normalized to [0, 1] -- ready for a CNN.

    This is an STFT-based spectrogram, not gwpy's true Q-transform (which
    needs the currently-unavailable gwpy/igwn-segments toolchain) -- good
    enough to see a chirp, but revisit with a real Q-transform once that
    dependency can be installed (see ml/README.md).
    """
    fs = 1.0 / dt
    white = whiten(strain, dt)
    filtered = bandpass(white, dt, *freq_range)

    center_idx = int(round((event_gps - gps_start) / dt))
    half_win = int(round(window_s / 2 / dt))
    start, end = center_idx - half_win, center_idx + half_win
    if start < 0 or end > len(filtered):
        raise ValueError("window extends outside the loaded strain segment")
    segment = filtered[start:end]

    nperseg = 256
    noverlap = nperseg - 8  # fine time resolution
    freqs, _times, sxx = signal.spectrogram(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)

    band = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    sxx = sxx[band]

    log_power = np.log10(sxx + 1e-30)

    freq_zoom = out_size[0] / log_power.shape[0]
    time_zoom = out_size[1] / log_power.shape[1]
    resized = zoom(log_power, (freq_zoom, time_zoom), order=1)

    lo, hi = np.percentile(resized, [1, 99])
    normalized = np.clip((resized - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return normalized.astype(np.float32)


def qscan_row_frequencies(
    fs: float,
    freq_range: tuple[float, float] = (10.0, 2048.0),
    n_rows: int = 140,
) -> np.ndarray:
    """Center frequency (Hz) of each make_qscan row; row 0 is the highest."""
    return np.geomspace(freq_range[0], min(freq_range[1], fs / 2), n_rows)[::-1]


def make_qscan(
    strain: np.ndarray,
    dt: float,
    gps_start: float,
    event_gps: float,
    window_s: float = 1.0,
    freq_range: tuple[float, float] = (10.0, 2048.0),
    qs: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0, 64.0),
    out_size: tuple[int, int] = (140, 170),
) -> np.ndarray:
    """Constant-Q "omega scan" of whitened strain around `event_gps`, laid out
    like the Gravity Spy training images so both classifier classes share one
    visual domain: log-spaced 10-2048 Hz rows with high frequency at row 0,
    1 s of time, 140x170, normalized energy (noise ~1) clipped to [0, 25.5]
    and scaled to [0, 1] -- the same range Gravity Spy's colorbar uses.

    Same algorithm as gwpy's q_transform: for each row frequency f, apply a
    bisquare window of half-width f*sqrt(11)/Q around +f in the frequency
    domain, inverse FFT for the analytic band-limited signal, take |.|^2,
    normalize by the row's median energy over the full segment. Several Qs
    are tried and the plane with the highest peak is kept. Like gwpy, each
    row's inverse FFT only spans its own band (shifted to baseband, which
    leaves |.|^2 unchanged), so narrow low-frequency rows are cheap.
    """
    fs = 1.0 / dt
    white = whiten(strain, dt)
    n = len(white)
    duration = n * dt
    df = 1.0 / duration
    data_fft = np.fft.rfft(white)

    t_start = event_gps - gps_start - window_s / 2
    if t_start < 0 or t_start + window_s > duration:
        raise ValueError("window extends outside the loaded strain segment")

    sub = 8  # envelope samples averaged per output column
    t_fine = t_start + (np.arange(out_size[1] * sub) + 0.5) * window_s / (out_size[1] * sub)
    min_len = 1 << int(np.ceil(np.log2(2 * duration * out_size[1] / window_s)))

    row_freqs = qscan_row_frequencies(fs, freq_range, out_size[0])

    best_plane, best_peak = None, -np.inf
    for q in qs:
        plane = np.empty(out_size)
        for i, f in enumerate(row_freqs):
            half_bw = f * np.sqrt(11) / q
            k = np.arange(max(1, int(np.ceil((f - half_bw) / df))),
                          min(n // 2, int(np.floor((f + half_bw) / df))) + 1)
            x = (k * df - f) / half_bw
            length = max(min_len, 1 << int(np.ceil(np.log2(2 * len(k)))))
            band = np.zeros(length, dtype=complex)
            band[:len(k)] = data_fft[k] * (1 - x**2) ** 2
            energy = np.abs(np.fft.ifft(band)) ** 2
            fine = np.interp(t_fine, np.arange(length) * duration / length, energy, period=duration)
            plane[i] = fine.reshape(out_size[1], sub).mean(axis=1) / np.median(energy)
        if plane.max() > best_peak:
            best_plane, best_peak = plane, plane.max()

    return (np.clip(best_plane, 0.0, 25.5) / 25.5).astype(np.float32)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="GWOSC strain HDF5 file")
    parser.add_argument("--event-gps", type=float, required=True)
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    strain, dt, gps_start = load_gwosc_strain(args.input)
    spec = make_spectrogram(strain, dt, gps_start, args.event_gps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, spec)

    png_path = out_dir / f"{stem}.png"
    plt.figure(figsize=(5, 5))
    plt.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
    plt.title(f"{stem}\nwhitened spectrogram around GPS {args.event_gps}")
    plt.xlabel("time bin")
    plt.ylabel("freq bin (20-500 Hz)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)

    print(f"Saved {npy_path} and {png_path}")
