"""Estimate chirp mass for classified real signals by matched filtering.

A bank of post-Newtonian templates -- the same idea LIGO's searches use,
heavily simplified: non-spinning TaylorF2 (frequency-domain stationary-phase
chirp, phase to 2PN order), inspiral only (cut at the ISCO frequency, no
merger/ringdown), one detector at a time. For each template the whitened,
glitch-gated strain is correlated against it over all arrival times (one
inverse FFT); the peak within +-0.1 s of the catalog GPS time is scored by
chi-square reweighted SNR, and the best (Mc, q) template is the estimate.

This measures the *detector-frame* chirp mass (1+z)*Mc_source -- the
waveform alone can't separate mass from redshift without the distance -- so
it is scored against published detector-frame values.

An estimate only means something when its SNR clears what the same search
finds in pure noise (~3,000 templates always match noise at SNR 5-6 somewhere,
with an essentially random winner). The threshold is measured, not tuned:
the loudest SNR found in signal-free windows 9 s after 40 events. Above it,
chirp mass is recovered to ~1-2% for Mc_det < 30 and ~6% for 30-60. Heavy
systems mostly fall below it -- their SNR is in the merger-ringdown the
templates lack. Mass ratio only enters the phase at 1PN and is never better
than the population median here, so outputs use that median instead. These
are point estimates, not Bayesian parameter estimation -- note in the writeup.

    python -m ml.mass_estimation.regression
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import scipy.fft
from scipy import signal

from ml.preprocessing.spectrogram import load_gwosc_strain

T_SUN = 4.925491025543576e-6  # G * M_sun / c^3: one solar mass as a time, in seconds

# The search runs in two stages (see matched_filter_masses): every chirp mass
# at two mass ratios first, then the best one's neighbors at all of these.
STAGE1_Q = np.array([1.0, 0.4])
FINE_Q = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15])

CHISQ_BINS = 8  # frequency bands in the chi-square check that a match looks like a real chirp
MAX_TEMPLATE_S = 12.0  # stays inside the ~15 s of data before each event

# Inputs from ml/data/fetch_all_events.py, and caches of this search's results.
MANIFEST = "data/processed/real_signal/real_signal_manifest.csv"
MATCHED_FILTER_CACHE = "data/processed/real_signal/matched_filter.csv"
BACKGROUND_CACHE = "data/processed/real_signal/matched_filter_background.csv"


def chirp_mass_grid(mc_min: float = 0.8, mc_max: float = 250.0, step_at_1msun: float = 6e-4,
                    max_step: float = 0.03) -> np.ndarray:
    """Chirp masses spaced in log Mc with step ~ Mc^(5/3): the number of
    in-band cycles scales as Mc^(-5/3), so a neutron-star merger's SNR peak is
    ~0.1% wide in Mc while a heavy black-hole pair's is tens of percent.
    """
    grid = [mc_min]
    while grid[-1] < mc_max:
        grid.append(grid[-1] * np.exp(min(step_at_1msun * grid[-1] ** (5 / 3), max_step)))
    return np.array(grid)


MC_GRID = chirp_mass_grid()


def start_frequency(mc_s: float, f_low: float, max_duration_s: float = MAX_TEMPLATE_S) -> float:
    """Highest of `f_low` and the frequency at which a leading-order chirp is
    `max_duration_s` from merger (tau = 5/256 (pi f)^(-8/3) Mc^(-5/3)).
    Longer templates would wrap around the circular FFT correlation.
    """
    f_dur = (5 / 256 * mc_s ** (-5 / 3) / max_duration_s) ** (3 / 8) / np.pi
    return max(f_low, f_dur)


def taylorf2_phase(f: np.ndarray, mc_s: float, eta: float) -> np.ndarray:
    """Non-spinning TaylorF2 phase to 2PN, with t_c = phi_c = 0."""
    v = (np.pi * mc_s * eta**-0.6 * f) ** (1 / 3)
    return 3 / (128 * eta * v**5) * (
        1
        + 20 / 9 * (743 / 336 + 11 / 4 * eta) * v**2
        - 16 * np.pi * v**3
        + 10 * (3058673 / 1016064 + 5429 / 1008 * eta + 617 / 144 * eta**2) * v**4
    )


def gate_mask(
    white: np.ndarray,
    dt: float,
    center_idx: int,
    threshold_sigma: float = 8.0,
    gate_s: float = 0.25,
    protect_s: float = 0.25,
) -> np.ndarray:
    """Multiplicative mask for whitened data that zeroes (with a 0.1 s taper)
    +-`gate_s` around any sample louder than `threshold_sigma` (robust sigma).
    Gaussian noise essentially never gets there; loud glitches do -- e.g. the
    one ~1 s before GW170817 in L1, which LIGO gated the same way. Samples
    within `protect_s` of the event are never gated, so a loud real merger
    can't remove itself. Must be applied to *whitened* data: zeroing raw strain
    (dominated by low-frequency noise) leaves step transients that whitening
    turns into new spikes.
    """
    sigma = 1.4826 * np.median(np.abs(white - np.median(white)))
    idx = np.arange(len(white))
    loud = (np.abs(white) > threshold_sigma * sigma) & (np.abs(idx - center_idx) > protect_s / dt)
    if not loud.any():
        return np.ones_like(white)
    half = int(gate_s / dt)
    gated = np.convolve(loud.astype(float), np.ones(2 * half + 1), mode="same") > 0
    taper = np.hanning(int(0.1 / dt))
    return 1 - np.clip(np.convolve(gated.astype(float), taper / taper.sum(), mode="same") * 2, 0, 1)


class _Filter:
    """Whitened, glitch-gated data for one strain segment; `snr` scores one template."""

    def __init__(self, strain: np.ndarray, dt: float, gps_start: float, event_gps: float,
                 f_low: float, search_s: float):
        fs = 1.0 / dt
        n = len(strain)
        center = int(round((event_gps - gps_start) / dt))
        psd_freqs, psd = signal.welch(strain, fs=fs, nperseg=int(4 * fs), average="median")
        self.freqs = np.fft.rfftfreq(n, dt)
        self.asd = np.sqrt(np.interp(self.freqs, psd_freqs, psd))
        self.asd[self.asd == 0] = np.inf

        white_fft = np.fft.rfft(strain * signal.windows.tukey(n, 0.1)) * dt / self.asd
        # Templates start at f_low, and below it a 4 s Welch PSD can't resolve
        # the huge seismic/near-DC power (GW150914's L1 data: whitened power
        # peaks under 0.5 Hz), which would otherwise trip the gate everywhere.
        white_fft[self.freqs < f_low] = 0
        white = np.fft.irfft(white_fft, n)
        mask = gate_mask(white, dt, center)
        self.white_fft = np.fft.rfft(white * mask)
        self.gated_fraction = float(np.mean(mask < 1))

        self.df = self.freqs[1]
        self.n = n
        self.f_low = f_low
        self.f_high = min(0.45 * fs, 1000.0)  # little inspiral SNR above 1 kHz; halves low-mass FFTs
        self.window = slice(center - int(search_s * fs), center + int(search_s * fs))

    def snr(self, mc: float, q: float) -> float:
        eta = q / (1 + q) ** 2
        mc_s = mc * T_SUN
        f_isco = 1 / (6**1.5 * np.pi * mc_s * eta**-0.6)
        f_start = start_frequency(mc_s, self.f_low)
        band = (self.freqs >= f_start) & (self.freqs < min(f_isco, self.f_high))
        if band.sum() < 16:
            return 0.0
        bins = np.flatnonzero(band)
        fb = self.freqs[band]
        h = fb ** (-7 / 6) * np.exp(-1j * taylorf2_phase(fb, mc_s, eta))
        terms = self.white_fft[band] * np.conj(h) / self.asd[band]
        power = np.abs(h) ** 2 / self.asd[band] ** 2
        sigma = np.sqrt(4 * self.df * power.sum())

        # A template that stops at f_max only needs an FFT whose Nyquist exceeds
        # f_max -- heavy black-hole templates end below ~100 Hz, so this cuts
        # their inverse FFT up to 32x; time resolution drops by the same factor.
        decim = 1
        while self.n % (2 * decim) == 0 and bins[-1] < self.n // (4 * decim):
            decim *= 2
        n_eff = self.n // decim
        integrand = np.zeros(n_eff, dtype=complex)
        integrand[bins] = terms
        z = 4 * self.df * n_eff * scipy.fft.ifft(integrand, workers=-1)
        lo, hi = self.window.start // decim, self.window.stop // decim
        peak = lo + int(np.abs(z[lo:hi]).argmax())
        rho = np.abs(z[peak]) / sigma
        return rho / self._chisq_penalty(terms * np.exp(2j * np.pi * bins * peak / n_eff), power, z[peak])

    def _chisq_penalty(self, shifted_terms: np.ndarray, power: np.ndarray, z_peak: complex) -> float:
        """Allen chi-square over CHISQ_BINS equal-power frequency bands, folded
        into LVK's reweighted SNR: divide by [(1 + chi2_r^3) / 2]^(1/6) when the
        reduced chi-square exceeds 1. A real chirp accumulates SNR evenly across
        bands (chi2_r ~ 1); a glitch dumps it into a few, so it's penalized.
        """
        edges = np.searchsorted(np.cumsum(power) / power.sum(), np.linspace(0, 1, CHISQ_BINS + 1)[1:-1])
        sigma_sq = 4 * self.df * power.sum()
        z_bins = np.array([4 * self.df * part.sum() for part in np.split(shifted_terms, edges)])
        chisq = CHISQ_BINS * np.sum(np.abs(z_bins - z_peak / CHISQ_BINS) ** 2) / sigma_sq
        chisq_r = chisq / (2 * CHISQ_BINS - 2)
        return 1.0 if chisq_r <= 1 else ((1 + chisq_r**3) / 2) ** (1 / 6)


def matched_filter_masses(
    strain: np.ndarray,
    dt: float,
    gps_start: float,
    event_gps: float,
    f_low: float = 20.0,
    search_s: float = 0.1,
) -> tuple[float, float, float]:
    """Returns (reweighted SNR, detector-frame chirp mass in M_sun, mass ratio
    <= 1) of the best-matching template. Stage 1 scans the full chirp-mass
    grid at two mass ratios (leading-order phase depends on Mc alone); stage 2
    rescans the neighboring grid points across the finer mass-ratio grid.
    """
    mf = _Filter(strain, dt, gps_start, event_gps, f_low, search_s)

    # Stage 1: (snr, grid index, mass ratio) of the best coarse match.
    best_snr, best_i, best_q = max((mf.snr(mc, q), i, q) for i, mc in enumerate(MC_GRID) for q in STAGE1_Q)

    # Stage 2: the 4 grid points either side, at every fine mass ratio.
    neighbors = MC_GRID[max(0, best_i - 4):best_i + 5]
    candidates = [(best_snr, MC_GRID[best_i], best_q)]
    candidates += [(mf.snr(mc, q), mc, q) for mc in neighbors for q in FINE_Q]
    snr, mc, q = max(candidates)
    return snr, float(mc), float(q)


def _run_cached(jobs: list[tuple[str, str, float, str]], strain_dir: Path, cache_path: Path) -> dict[str, dict]:
    """Matched-filters each (key, strain stem, center GPS, label) job, caching
    results by key in `cache_path` so interrupted runs resume.
    """
    cached: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, newline="") as f:
            cached = {r["stem"]: r for r in csv.DictReader(f)}

    todo = [job for job in jobs if job[0] not in cached]
    for i, (key, stem, center_gps, label) in enumerate(todo, 1):
        strain, dt, gps_start = load_gwosc_strain(str(strain_dir / f"{stem}.hdf5"))
        snr, mc, q = matched_filter_masses(strain, dt, gps_start, center_gps)
        cached[key] = {"stem": key, "snr": snr, "mc_raw": mc, "q_raw": q}
        print(f"[{i}/{len(todo)}] {label}: SNR {snr:.1f}  Mc_det {mc:.2f}  q {q:.2f}")
        with open(cache_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["stem", "snr", "mc_raw", "q_raw"])
            writer.writeheader()
            writer.writerows(cached.values())
    return cached


def noise_background(manifest_rows: list[dict], strain_dir: Path, n_segments: int = 40,
                     offset_s: float = 9.0) -> np.ndarray:
    """Best reweighted SNR the full template search finds in signal-free data:
    the same search, centered `offset_s` after each of `n_segments` events.
    With ~3,000 templates, pure noise routinely produces SNR 5-6 somewhere in
    the bank, and the template that wins is essentially random -- so an
    estimate only means something when its SNR clears this background.
    """
    rows = manifest_rows[:: max(1, len(manifest_rows) // n_segments)][:n_segments]
    jobs = [(f"{r['stem']}__off+{offset_s:g}", r["stem"], float(r["gps"]) + offset_s, f"noise {r['event']}/{r['detector']}")
            for r in rows]
    results = _run_cached(jobs, strain_dir, Path(BACKGROUND_CACHE))
    return np.array([float(results[key]["snr"]) for key, *_ in jobs])


def _best_detector_per_event(manifest_rows: list[dict], results: dict[str, dict]) -> list[dict]:
    """One row per event, from whichever detector had the higher matched-filter SNR."""
    by_event: dict[str, dict] = {}
    for row in manifest_rows:
        r = results[row["stem"]]
        merged = {**row, "snr": float(r["snr"]), "mc_raw": float(r["mc_raw"]), "q_raw": float(r["q_raw"])}
        if row["event"] not in by_event or merged["snr"] > by_event[row["event"]]["snr"]:
            by_event[row["event"]] = merged
    return list(by_event.values())


def _classifier_real_signal_events(manifest_rows: list[dict], qscan_dir: Path, checkpoint: Path) -> set[str]:
    """Events where the trained classifier calls at least one detector's Q-scan real_signal."""
    import torch

    from ml.classifier.dataset import REAL_SIGNAL_LABEL, prepare_image
    from ml.classifier.model import build_model

    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    real_idx = ckpt["classes"].index(REAL_SIGNAL_LABEL)

    passed = set()
    with torch.no_grad():
        for row in manifest_rows:
            image = prepare_image(np.load(qscan_dir / f"{row['stem']}.npy"), tuple(ckpt["image_size"]))
            if model(torch.from_numpy(image)[None, None]).argmax(dim=1).item() == real_idx:
                passed.add(row["event"])
    return passed


def _save_scatter(true: np.ndarray, est: np.ndarray, measurable: np.ndarray, threshold: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    lims = (0.7, 300)
    ax.plot(lims, lims, "k--", lw=1, label="perfect")
    ax.scatter(true[~measurable], est[~measurable], s=12, alpha=0.35, color="gray",
               label=f"SNR <= {threshold:.1f} (noise background)")
    ax.scatter(true[measurable], est[measurable], s=16, alpha=0.85, label=f"SNR > {threshold:.1f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("published detector-frame chirp mass (M_sun)")
    ax.set_ylabel("matched-filter chirp mass (M_sun)")
    ax.legend(loc="upper left")
    ax.set_title("Chirp mass: TaylorF2 matched filter vs. GWTC")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(
    manifest_path: str = MANIFEST,
    strain_dir: str = "data/raw/strain",
    qscan_dir: str = "data/processed/real_signal",
    checkpoint: str = "data/checkpoints/classifier_best.pt",
    out_csv: str = "data/mass_estimates.csv",
    report_dir: str = "data/reports",
) -> None:
    with open(manifest_path, newline="") as f:
        manifest = [r for r in csv.DictReader(f) if r["chirp_mass_det_msun"] and r["mass_ratio"]]

    jobs = [(r["stem"], r["stem"], float(r["gps"]), f"{r['event']}/{r['detector']}") for r in manifest]
    events = _best_detector_per_event(manifest, _run_cached(jobs, Path(strain_dir), Path(MATCHED_FILTER_CACHE)))
    background = noise_background(manifest, Path(strain_dir))
    threshold = float(background.max())

    true_mc = np.array([float(e["chirp_mass_det_msun"]) for e in events])
    true_q = np.array([float(e["mass_ratio"]) for e in events])
    est_mc = np.array([e["mc_raw"] for e in events])
    est_q = np.array([e["q_raw"] for e in events])
    snr = np.array([e["snr"] for e in events])
    measurable = snr > threshold
    rel = np.abs(est_mc / true_mc - 1)
    prior_q = float(np.median(true_q))

    def stats(mask: np.ndarray) -> str:
        return (f"n={mask.sum():3d}  median |rel err| {np.median(rel[mask]):.3f}  "
                f"within 10% {np.mean(rel[mask] < 0.1):.2f}  within 25% {np.mean(rel[mask] < 0.25):.2f}")

    lines = [
        f"Matched-filter chirp mass on {len(events)} events (best-SNR detector each)",
        f"noise background: best SNR over the bank in {len(background)} signal-free windows: "
        f"median {np.median(background):.1f}, max {threshold:.1f} -> estimates need SNR > {threshold:.1f}",
        "",
        f"  all events                    {stats(np.ones_like(measurable))}",
        f"  SNR > {threshold:4.1f} (measurable)     {stats(measurable)}",
        f"  SNR <= {threshold:4.1f} (noise-like)    {stats(~measurable)}",
        f"  predict-the-median baseline   median |rel err| {np.median(np.abs(np.median(true_mc) / true_mc - 1)):.3f}",
        "",
        "measurable events, by published Mc_det:",
    ]
    for lo, hi in [(0, 15), (15, 30), (30, 60), (60, np.inf)]:
        s = measurable & (true_mc >= lo) & (true_mc < hi)
        lines.append(f"  {lo:>3}-{hi:<4} " + (stats(s) if s.any() else "n=  0"))
    lines += [
        f"heavy systems (published Mc_det >= 60) clearing the threshold: "
        f"{(measurable & (true_mc >= 60)).sum()}/{(true_mc >= 60).sum()} -- inspiral-only templates "
        "miss the merger-ringdown that carries most of their SNR",
        "",
        f"mass ratio (measurable events): matched filter MAE {np.mean(np.abs(est_q - true_q)[measurable]):.3f} vs. "
        f"population median ({prior_q:.2f}) MAE {np.mean(np.abs(prior_q - true_q)[measurable]):.3f} -- "
        "not measurable here; outputs use the population median",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mass_estimation_report.txt").write_text(report + "\n")
    _save_scatter(true_mc, est_mc, measurable, threshold, out_dir / "mass_estimation_scatter.png")

    if Path(checkpoint).exists():
        classified_real = _classifier_real_signal_events(manifest, Path(qscan_dir), Path(checkpoint))
    else:
        print(f"no classifier checkpoint at {checkpoint}; skipping classifier gating")
        classified_real = {e["event"] for e in events}

    written = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "chirp_mass_msun", "mass_ratio", "mass_ratio_source", "detector", "snr",
                         "published_chirp_mass_msun", "published_mass_ratio"])
        for e, ok in sorted(zip(events, measurable), key=lambda pair: float(pair[0]["gps"])):
            if ok and e["event"] in classified_real:
                writer.writerow([e["event"], f"{e['mc_raw']:.3f}", f"{prior_q:.3f}", "population median (not measured)",
                                 e["detector"], f"{e['snr']:.2f}", f"{float(e['chirp_mass_det_msun']):.3f}",
                                 f"{float(e['mass_ratio']):.3f}"])
                written += 1
    print(f"\nclassifier calls {len(classified_real)}/{len(events)} events real_signal; "
          f"{written} of those clear the SNR threshold and go to {out_csv}")
    print(f"Saved {out_csv}, {out_dir / 'mass_estimation_report.txt'}, {out_dir / 'mass_estimation_scatter.png'}")


if __name__ == "__main__":
    run()
