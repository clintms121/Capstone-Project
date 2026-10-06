"""Injection-recovery test: how loud does a chirp have to be for the
classifier to catch it, and how often does it flag plain noise?

The held-out test score can't answer this on its own: the first model scored
a perfect test set while calling all pure noise real_signal, and even after
adding off-source noise, noise within ~3 s of a loud event is flagged far
more often than noise farther away. So, as LIGO measures search sensitivity:
simulated chirps (non-spinning TaylorF2 inspirals, the same templates as
ml/mass_estimation) are added to real detector noise at known optimal SNR,
rendered with the same Q-scan, and classified. Noise comes from test-split
strain files only, 8-10 s after their event (no signal in band, never seen
in training), so neither the noise nor the event is something the model
trained on. Plain-noise windows at the same times give the false-alarm rate.

Injected chirp masses are 8-35 M_sun (detector frame): the templates are
inspiral-only, and heavier systems put most of their visible energy in the
merger and ringdown they lack. Optimal SNR is computed for the template up to
min(ISCO, 1 kHz) against the segment's own PSD, the same scale as the
matched-filter SNRs in ml/mass_estimation -- not the published network SNR.

    python -m ml.classifier.injection_test
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from scipy import signal

from ml.classifier.dataset import REAL_SIGNAL_LABEL, load_dataset, prepare_image, split_dataset
from ml.classifier.model import build_model
from ml.mass_estimation.regression import T_SUN, taylorf2_phase
from ml.preprocessing.spectrogram import load_gwosc_strain, make_qscan

SNR_LEVELS = (4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)


def make_injection(
    strain: np.ndarray, dt: float, t_coalesce: float, chirp_mass_det: float, mass_ratio: float,
    target_snr: float, f_low: float = 15.0,
) -> np.ndarray:
    """Time-domain TaylorF2 inspiral coalescing `t_coalesce` s into the
    segment, scaled to `target_snr` against the segment's median-Welch PSD.
    Band edges are tapered so the template doesn't start or stop with a click.
    """
    n = len(strain)
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(n, dt)
    psd_freqs, psd = signal.welch(strain, fs=fs, nperseg=int(4 * fs), average="median")
    psd = np.interp(freqs, psd_freqs, psd)

    eta = mass_ratio / (1 + mass_ratio) ** 2
    mc_s = chirp_mass_det * T_SUN
    f_high = min(1 / (6**1.5 * np.pi * mc_s * eta**-0.6), 1000.0)
    band = (freqs >= f_low) & (freqs < f_high)
    fb = freqs[band]
    taper = np.clip((fb - f_low) / 7.0, 0, 1) * np.clip((f_high - fb) / (0.1 * f_high), 0, 1)

    h = np.zeros(len(freqs), dtype=complex)
    h[band] = taper * fb ** (-7 / 6) * np.exp(-1j * (taylorf2_phase(fb, mc_s, eta) + 2 * np.pi * fb * t_coalesce))
    snr = np.sqrt(4 * freqs[1] * np.sum(np.abs(h[band]) ** 2 / psd[band]))
    return np.fft.irfft(h * target_snr / snr, n) / dt


def run(
    checkpoint: str = "data/checkpoints/classifier_best.pt",
    manifest: str = "data/processed/real_signal/real_signal_manifest.csv",
    strain_dir: str = "data/raw/strain",
    report_dir: str = "data/reports",
    seed: int = 0,
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    real_idx = ckpt["classes"].index(REAL_SIGNAL_LABEL)

    _, _, test = split_dataset(load_dataset(), seed=seed)
    test_groups = {s.group_id for s in test if s.source == "qscan"}
    with open(manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["stem"].split("-")[-2] in test_groups]

    rng = np.random.default_rng(seed)
    images, snr_of = [], []
    for r in rows:
        strain, dt, gps_start = load_gwosc_strain(f"{strain_dir}/{r['stem']}.hdf5")
        if not np.isfinite(strain).all():
            continue
        t_c = float(r["gps"]) - gps_start + rng.uniform(8.0, 10.0)
        images.append(make_qscan(strain, dt, gps_start, gps_start + t_c))
        snr_of.append(0.0)
        for target in SNR_LEVELS:
            mc = float(np.exp(rng.uniform(np.log(8.0), np.log(35.0))))
            q = float(rng.uniform(0.3, 1.0))
            injected = strain + make_injection(strain, dt, t_c, mc, q, target)
            images.append(make_qscan(injected, dt, gps_start, gps_start + t_c))
            snr_of.append(target)
        print(f"{r['event']}/{r['detector']} done")

    x = torch.from_numpy(np.stack([prepare_image(i, ckpt["image_size"]) for i in images]))[:, None]
    with torch.no_grad():
        called = (model(x).argmax(1) == real_idx).numpy()
    snr_of = np.array(snr_of)

    lines = [f"Injection test on {len(rows)} test-split detector segments (noise 8-10 s after each event)",
             f"  false alarms on plain noise: {called[snr_of == 0].sum()}/{(snr_of == 0).sum()}"
             f" ({called[snr_of == 0].mean():.1%})",
             "  injected TaylorF2 chirps (Mc_det 8-35, q 0.3-1) caught, by optimal SNR:"]
    efficiency = []
    for target in SNR_LEVELS:
        m = snr_of == target
        efficiency.append(called[m].mean())
        lines.append(f"    SNR {target:4.0f}: {called[m].sum():3d}/{m.sum():3d}  ({called[m].mean():.0%})")
    report = "\n".join(lines)
    print("\n" + report)

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "injection_test_report.txt").write_text(report + "\n")
    with open(out / "injection_efficiency.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr", "caught", "n"])  # snr 0 = plain noise (false alarms)
        for level in (0.0,) + SNR_LEVELS:
            m = snr_of == level
            writer.writerow([level, int(called[m].sum()), int(m.sum())])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SNR_LEVELS, efficiency, "o-", label="injected chirps called real_signal")
    ax.axhline(called[snr_of == 0].mean(), color="r", ls="--", label="false-alarm rate (plain noise)")
    ax.set_xlabel("injected optimal SNR (single detector)")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center right")
    ax.set_title("Classifier detection efficiency")
    fig.tight_layout()
    fig.savefig(out / "injection_efficiency.png", dpi=130)
    plt.close(fig)
    print(f"Saved {out / 'injection_test_report.txt'}, {out / 'injection_efficiency.png'}")


if __name__ == "__main__":
    run()
