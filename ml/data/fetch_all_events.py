"""Batch-fetch confirmed GWOSC events and turn each detector's strain into a
real-signal Q-scan for ml/classifier/dataset.py.

For every event passing the filters below, downloads the 32 s / 4 kHz strain
file per detector (ml/data/fetch_ligo_strain.py), runs it through
ml/preprocessing/spectrogram.py:make_qscan (Gravity Spy image layout), and
saves data/processed/real_signal/<strain stem>.npy. Also writes
data/processed/real_signal/real_signal_manifest.csv with each spectrogram's
event name and published parameters (chirp mass, mass ratio) -- the
regression targets ml/mass_estimation will need.

Also saves off-source noise Q-scans from the same files (seconds away from
the event) to data/processed/noise/<strain stem>__off<offset>.npy. Without
them, every image my Q-scan pipeline produced was a real signal, and the
classifier learned "rendered by my pipeline -> real_signal" instead of
"contains a chirp" (it called 100% of pure noise real). Low-mass systems,
whose inspiral is still in band seconds before merger, only get post-merger
noise.

Filters, so nothing that might be noise gets the real_signal label:
- LVK confident catalogs only (no marginal/auxiliary candidates, no
  third-party IAS search, no Initial LIGO entries)
- p_astro >= 0.5 where reported
- network matched-filter SNR >= --min-snr (default 10): quieter events often
  show no visible chirp in a single detector's spectrogram, which would teach
  the classifier that plain noise is a real signal
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ml.data.fetch_ligo_strain import fetch_event_strain, list_confident_events
from ml.preprocessing.spectrogram import load_gwosc_strain, make_qscan

EXCLUDED_CATALOGS = {"IAS-O3a", "Initial_LIGO_Virgo"}
NOISE_OFFSETS_S = (-12.0, -6.0, 6.0, 12.0)
LONG_INSPIRAL_CHIRP_MASS = 5.0  # detector-frame M_sun; below this, pre-merger data isn't signal-free

MANIFEST_FIELDS = [
    "stem", "event", "detector", "catalog", "gps", "network_snr", "p_astro",
    "chirp_mass_msun", "chirp_mass_det_msun", "mass_ratio",
]


def select_events(min_snr: float) -> list[dict]:
    selected = []
    for e in list_confident_events():
        catalog = e["catalog.shortName"]
        if catalog in EXCLUDED_CATALOGS or "marginal" in catalog or "auxiliary" in catalog:
            continue
        if e["p_astro"] is not None and e["p_astro"] < 0.5:
            continue
        if e["network_matched_filter_snr"] is None or e["network_matched_filter_snr"] < min_snr:
            continue
        selected.append(e)
    return sorted(selected, key=lambda e: e["GPS"])


def _detector_frame_chirp_mass(event: dict) -> float | None:
    """Redshifted chirp mass (1+z)*Mc_source -- what a waveform in the
    detector actually encodes. GWOSC's own chirp_mass field is left empty.
    """
    if event["chirp_mass_source"] is None or event["redshift"] is None:
        return None
    return event["chirp_mass_source"] * (1 + event["redshift"])


def _mass_ratio(event: dict) -> float | None:
    m1, m2 = event["mass_1_source"], event["mass_2_source"]
    if m1 is None or m2 is None:
        return None
    return min(m1, m2) / max(m1, m2)


def _noise_offsets(event: dict) -> tuple[float, ...]:
    mc = _detector_frame_chirp_mass(event)
    if mc is None or mc < LONG_INSPIRAL_CHIRP_MASS:
        return tuple(o for o in NOISE_OFFSETS_S if o > 0)
    return NOISE_OFFSETS_S


def fetch_all_events(
    detectors: tuple[str, ...] = ("H1", "L1"),
    min_snr: float = 10.0,
    out_dir: str = "data/processed/real_signal",
    noise_dir: str = "data/processed/noise",
) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    noise_out = Path(noise_dir)
    noise_out.mkdir(parents=True, exist_ok=True)

    events = select_events(min_snr)
    print(f"{len(events)} events pass the filters; fetching {', '.join(detectors)}")

    rows = []
    for i, event in enumerate(events, 1):
        name = event["commonName"]
        for det in detectors:
            try:
                strain_path, gps = fetch_event_strain(name, det)
                targets = {out / f"{strain_path.stem}.npy": gps}
                targets.update({noise_out / f"{strain_path.stem}__off{off:+g}.npy": gps + off
                                for off in _noise_offsets(event)})
                missing = {path: t for path, t in targets.items() if not path.exists()}
                if missing:
                    strain, dt, gps_start = load_gwosc_strain(str(strain_path))
                    if not np.isfinite(strain).all():
                        raise ValueError("strain has NaN gaps (detector data-quality dropout)")
                    for path, center_gps in missing.items():
                        np.save(path, make_qscan(strain, dt, gps_start, center_gps))
            except Exception as exc:
                print(f"[{i}/{len(events)}] skip {name}/{det}: {exc}")
                continue

            rows.append({
                "stem": strain_path.stem,
                "event": name,
                "detector": det,
                "catalog": event["catalog.shortName"],
                "gps": gps,
                "network_snr": event["network_matched_filter_snr"],
                "p_astro": event["p_astro"],
                "chirp_mass_msun": event["chirp_mass_source"],
                "chirp_mass_det_msun": _detector_frame_chirp_mass(event),
                "mass_ratio": _mass_ratio(event),
            })
            print(f"[{i}/{len(events)}] {name}/{det} ok")

    with open(out / "real_signal_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    parser.add_argument("--min-snr", type=float, default=10.0)
    parser.add_argument("--out-dir", default="data/processed/real_signal")
    parser.add_argument("--noise-dir", default="data/processed/noise")
    args = parser.parse_args()

    rows = fetch_all_events(tuple(args.detectors), args.min_snr, args.out_dir, args.noise_dir)
    n_events = len({r["event"] for r in rows})
    print(f"Saved {len(rows)} spectrograms from {n_events} events to {args.out_dir}")
