"""Export everything the C++ app's Classifier screen shows.

The trained models run in Python, so the app reads their results from files
in formats C++ can parse without extra libraries:

    data/demo/samples.csv     one row per demo sample: what it is, which split
                              it was in, the classifier's top-3 labels and
                              probabilities, and (real mergers) the matched-
                              filter chirp mass next to the published value
                              and how far away the merger was
    data/demo/img/<id>.pgm    the Q-scan the classifier saw (8-bit grayscale)
    data/demo/metrics.csv     headline numbers on the held-out test split
    data/demo/injection.csv   detection efficiency vs. injected SNR

Samples: every real-merger recording (marked held out or used in training),
plus held-out test-split Gravity Spy glitches (up to 3 per class) and
held-out off-source noise, so the demo shows the classifier on all three
kinds of input -- and every held-out noise image the classifier mistook for
a merger (kind "false_positive"). Run after training, the injection test and
mass estimation:

    python -m ml.export_demo
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from ml.classifier.dataset import (
    NOISE_LABEL, REAL_SIGNAL_LABEL, _event_group_id, load_dataset, prepare_image, split_dataset,
)
from ml.classifier.model import build_model
from ml.data.fetch_ligo_strain import list_confident_events

OUT = Path("data/demo")
MANIFEST = "data/processed/real_signal/real_signal_manifest.csv"
NOISE_DIR = Path("data/processed/noise")
MATCHED_FILTER = "data/processed/real_signal/matched_filter.csv"
BACKGROUND = "data/processed/real_signal/matched_filter_background.csv"
GLITCHES_PER_CLASS = 3
NOISE_SAMPLES = 12

FIELDS = ["id", "kind", "name", "detector", "true_label", "split", "pred1", "p1", "pred2", "p2", "pred3", "p3",
          "p_real", "snr", "mc_est", "mc_pub", "measurable", "distance_ly", "image"]

# Planck 2015 flat Lambda-CDM, the cosmology LVK catalogs use for redshift.
H0_KM_S_MPC = 67.74
OMEGA_M = 0.3075
HUBBLE_TIME_YR = 977.8e9 / H0_KM_S_MPC  # 1/H0 in years


def _write_pgm(image: np.ndarray, path: Path) -> None:
    h, w = image.shape
    pixels = np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
    path.write_bytes(f"P5\n{w} {h}\n255\n".encode() + pixels.tobytes())


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def light_travel_distance_ly(z: float) -> float:
    """How far the waves traveled to reach Earth, in light-years: lookback time
    to redshift z times c. This is the "1.3 billion light-years" LIGO quotes
    for GW150914; the luminosity distance in the catalog is larger (440+ Mpc
    = 1.4+ billion ly) and grows much faster with z.
    """
    zs = np.linspace(0.0, z, 2001)
    return HUBBLE_TIME_YR * np.trapezoid(1.0 / ((1 + zs) * np.sqrt(OMEGA_M * (1 + zs) ** 3 + 1 - OMEGA_M)), zs)


def _event_distances_ly() -> dict[str, float]:
    """Light-travel distance per event name from GWOSC's published redshift
    (one request to the event catalog). Events without a redshift, or every
    event if GWOSC can't be reached, are left out and shown without one.
    """
    try:
        events = list_confident_events()
    except OSError as e:
        print(f"couldn't reach GWOSC ({e}); exporting without distances")
        return {}
    return {e["commonName"]: light_travel_distance_ly(e["redshift"]) for e in events if e["redshift"] is not None}


def main() -> None:
    ckpt = torch.load("data/checkpoints/classifier_best.pt", map_location="cpu")
    classes, size = ckpt["classes"], tuple(ckpt["image_size"])
    model = build_model(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def predict(images: list[np.ndarray]) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for start in range(0, len(images), 256):
                x = torch.from_numpy(np.stack(images[start:start + 256]))[:, None]
                probs.append(torch.softmax(model(x), dim=1).numpy())
        return np.concatenate(probs)

    print("loading dataset and reproducing the training split...")
    samples = load_dataset()
    train, val, test = split_dataset(samples, seed=0)
    split_of = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        split_of.update({s.group_id: name for s in part})

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "img").mkdir(parents=True)

    mf = {r["stem"]: r for r in _read_csv(MATCHED_FILTER)}
    threshold = max(float(r["snr"]) for r in _read_csv(BACKGROUND))

    distances = _event_distances_ly()
    manifest = _read_csv(MANIFEST)
    rows, images = [], []
    for r in manifest:
        group = _event_group_id(Path(r["stem"]))
        m = mf.get(r["stem"])
        snr = float(m["snr"]) if m else float("nan")
        rows.append({
            "kind": "event", "name": r["event"], "detector": r["detector"], "true_label": REAL_SIGNAL_LABEL,
            "split": split_of.get(group, "train"), "snr": snr,
            "mc_est": float(m["mc_raw"]) if m else float("nan"),
            "mc_pub": float(r["chirp_mass_det_msun"]) if r["chirp_mass_det_msun"] else float("nan"),
            "measurable": int(bool(m) and snr > threshold),
            "distance_ly": distances.get(r["event"], float("nan")),
        })
        images.append(prepare_image(np.load(f"data/processed/real_signal/{r['stem']}.npy"), size))

    per_class = defaultdict(list)
    for s in sorted(test, key=lambda s: s.group_id):
        if s.source == "gravityspy" and len(per_class[s.label]) < GLITCHES_PER_CLASS:
            per_class[s.label].append(s)
    noise = [s for s in sorted(test, key=lambda s: s.group_id) if s.source == "qscan" and s.label == NOISE_LABEL]
    picks = [(s, "glitch") for label in sorted(per_class) for s in per_class[label]]
    picks += [(s, "noise") for s in noise[:: max(1, len(noise) // NOISE_SAMPLES)][:NOISE_SAMPLES]]
    for s, kind in picks:
        name = "Hardware-injected chirp" if s.label == REAL_SIGNAL_LABEL else s.label.replace("_", " ")
        if kind == "noise":
            name = "Detector noise"
        rows.append({"kind": kind, "name": f"{name} ({s.group_id})", "detector": "", "true_label": s.label,
                     "split": "test", "snr": float("nan"), "mc_est": float("nan"), "mc_pub": float("nan"),
                     "measurable": 0, "distance_ly": float("nan")})
        images.append(s.image)

    # Held-out off-source noise the classifier calls a merger: the false
    # positives behind the test split's real_signal precision. Loaded straight
    # from the files (same scaling as the dataset) to keep the offset and event.
    real_idx = classes.index(REAL_SIGNAL_LABEL)
    event_of = {r["stem"]: r for r in manifest}
    noise_paths = sorted(p for p in NOISE_DIR.glob("*.npy") if split_of.get(_event_group_id(p)) == "test")
    noise_images = [prepare_image(np.load(p), size) for p in noise_paths]
    for path, image, p in zip(noise_paths, noise_images, predict(noise_images)):
        if p.argmax() != real_idx:
            continue
        stem, offset = path.stem.split("__off")
        event = event_of[stem]
        when = f"{abs(float(offset)):g} s {'after' if float(offset) > 0 else 'before'}"
        rows.append({"kind": "false_positive", "name": f"Noise {when} {event['event']}",
                     "detector": event["detector"], "true_label": NOISE_LABEL, "split": "test",
                     "snr": float("nan"), "mc_est": float("nan"), "mc_pub": float("nan"), "measurable": 0,
                     "distance_ly": float("nan")})
        images.append(image)

    probs = predict(images)
    with open(OUT / "samples.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, (row, image, p) in enumerate(zip(rows, images, probs)):
            top = np.argsort(p)[::-1][:3]
            image_path = OUT / "img" / f"{i:04d}.pgm"
            _write_pgm(image, image_path)
            writer.writerow({
                **row, "id": i,
                "pred1": classes[top[0]], "p1": f"{p[top[0]]:.4f}",
                "pred2": classes[top[1]], "p2": f"{p[top[1]]:.4f}",
                "pred3": classes[top[2]], "p3": f"{p[top[2]]:.4f}",
                "p_real": f"{p[real_idx]:.4f}",
                "snr": f"{row['snr']:.2f}", "mc_est": f"{row['mc_est']:.3f}", "mc_pub": f"{row['mc_pub']:.3f}",
                "distance_ly": f"{row['distance_ly']:.4g}",
                "image": image_path.as_posix(),
            })

    print("scoring the held-out test split...")
    label_idx = {c: i for i, c in enumerate(classes)}
    test_probs = predict([s.image for s in test])
    y_true = np.array([label_idx[s.label] for s in test])
    y_pred = test_probs.argmax(axis=1)
    is_real, called_real = y_true == real_idx, y_pred == real_idx
    ours_noise = np.array([s.source == "qscan" and s.label == NOISE_LABEL for s in test])

    events = [r for r in rows if r["kind"] == "event"]
    best = {}
    for r in events:
        if not np.isnan(r["mc_pub"]) and (r["name"] not in best or r["snr"] > best[r["name"]]["snr"]):
            best[r["name"]] = r
    measured = [r for r in best.values() if r["snr"] > threshold]
    rel = [abs(r["mc_est"] / r["mc_pub"] - 1) for r in measured]

    metrics = {
        "test_images": len(test),
        "classes": len(classes),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "real_recall": float((is_real & called_real).sum() / is_real.sum()),
        "real_precision": float((is_real & called_real).sum() / max(called_real.sum(), 1)),
        "noise_called_real": int(called_real[ours_noise].sum()),
        "noise_total": int(ours_noise.sum()),
        "mass_threshold_snr": threshold,
        "mass_events": len(best),
        "mass_measurable": len(measured),
        "mass_median_error": float(np.median(rel)),
        "mass_within_10pct": float(np.mean(np.array(rel) < 0.1)),
    }
    with open(OUT / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(metrics.items())

    shutil.copy("data/reports/injection_efficiency.csv", OUT / "injection.csv")

    print(f"wrote {len(rows)} samples to {OUT}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
