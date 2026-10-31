"""Spectrogram dataset: Gravity Spy glitch classes + a "real signal" class.

Loads the real-signal Q-scans ml/data/fetch_all_events.py wrote to
data/processed/real_signal/ (one per event per detector) and the Gravity Spy
PNGs ml/data/fetch_gravityspy.py wrote to data/raw/gravityspy/<label>/, and
splits by group id so no near-duplicate image -- the same event seen from
multiple detectors -- leaks across train/val/test.

Both sources share one image layout (ml/preprocessing/spectrogram.py:
make_qscan mimics Gravity Spy's omega scans: 140x170, log 10-2048 Hz, high
frequency at row 0, normalized energy), and every image gets the same
per-image min-max scaling. Without that, the CNN could tell the classes apart
from rendering differences alone instead of morphology.

Gravity Spy's "Chirp" class is hardware-injected compact-binary signals --
physically the same morphology as real_signal -- so it's merged into it
rather than kept as a class the model can only separate by provenance.

Matching layouts wasn't enough on its own: trained on that alone, the model
called 100% of pure noise run through my Q-scan "real_signal", because my
pipeline only ever produced real signals. Off-source noise from the same
strain files (data/processed/noise/) is therefore added as No_Glitch --
Gravity Spy's own noise-only class -- grouped with its event so a segment's
signal and noise always land in the same split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

REAL_SIGNAL_LABEL = "real_signal"
NOISE_LABEL = "No_Glitch"
GLITCH_LABELS_MERGED_INTO_REAL_SIGNAL = {"Chirp"}


@dataclass
class Sample:
    image: np.ndarray  # (H, W) float32 in [0, 1]
    label: str
    group_id: str  # samples sharing a group_id never split across train/val/test
    source: str  # "qscan" (my pipeline) or "gravityspy"


def prepare_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image.shape != size:
        image = zoom(image, (size[0] / image.shape[0], size[1] / image.shape[1]), order=1)
    lo, hi = image.min(), image.max()
    return ((image - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _event_group_id(path: Path) -> str:
    """Groups same-event files across detectors -- e.g. H-H1_..-1187529241-32
    and L-L1_..-1187529241-32 (same analysis segment, different detector)
    share group id "1187529241" -- along with off-source noise cut from them
    (H-H1_..-1187529241-32__off-12).
    """
    match = re.search(r"-(\d+)-\d+$", path.stem.split("__")[0])
    return match.group(1) if match else path.stem


def _load_qscan_samples(qscan_dir: Path, label: str, size: tuple[int, int]) -> list[Sample]:
    return [
        Sample(prepare_image(np.load(npy_path), size), label, _event_group_id(npy_path), "qscan")
        for npy_path in sorted(qscan_dir.glob("*.npy"))
    ]


def _load_glitch_samples(glitch_dir: Path, size: tuple[int, int]) -> list[Sample]:
    import matplotlib.image as mpimg

    samples = []
    label_dirs = sorted(p for p in glitch_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for label_dir in label_dirs:
        label = REAL_SIGNAL_LABEL if label_dir.name in GLITCH_LABELS_MERGED_INTO_REAL_SIGNAL else label_dir.name
        for png_path in sorted(label_dir.glob("*.png")):
            raw = mpimg.imread(png_path)
            grayscale = raw[..., 0] if raw.ndim == 3 else raw
            samples.append(Sample(prepare_image(grayscale, size), label, png_path.stem, "gravityspy"))
    return samples


def load_dataset(
    real_signal_dir: str = "data/processed/real_signal",
    noise_dir: str = "data/processed/noise",
    glitch_dir: str = "data/raw/gravityspy",
    image_size: tuple[int, int] = (140, 170),
) -> list[Sample]:
    """Loads every real-signal Q-scan, off-source noise Q-scan and Gravity Spy
    image, min-max scaled (and resized, if a source doesn't match
    `image_size`), as a flat list of Sample.
    """
    samples = _load_qscan_samples(Path(real_signal_dir), REAL_SIGNAL_LABEL, image_size)
    samples += _load_qscan_samples(Path(noise_dir), NOISE_LABEL, image_size)
    samples += _load_glitch_samples(Path(glitch_dir), image_size)
    if not samples:
        raise RuntimeError(
            f"No samples found under {real_signal_dir!r} or {glitch_dir!r} -- "
            "run ml/data/fetch_all_events.py and ml/data/fetch_gravityspy.py first"
        )
    return samples


def class_names(samples: list[Sample]) -> list[str]:
    return sorted({s.label for s in samples})


def split_dataset(
    samples: list[Sample],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Splits by group_id -- never splitting the same event/glitch id across
    train/val/test -- stratified per class so rare glitch classes still show
    up in every split.
    """
    rng = np.random.default_rng(seed)

    label_by_group: dict[str, str] = {}
    for s in samples:
        label_by_group.setdefault(s.group_id, s.label)

    groups_by_label: dict[str, list[str]] = {}
    for group_id, label in label_by_group.items():
        groups_by_label.setdefault(label, []).append(group_id)

    train_groups: set[str] = set()
    val_groups: set[str] = set()
    test_groups: set[str] = set()

    for label, groups in groups_by_label.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * test_frac)) if n > 2 else 0
        n_val = max(1, round(n * val_frac)) if n > 1 else 0
        n_val = min(n_val, max(0, n - n_test))
        test_groups.update(groups[:n_test])
        val_groups.update(groups[n_test:n_test + n_val])
        train_groups.update(groups[n_test + n_val:])

    train = [s for s in samples if s.group_id in train_groups]
    val = [s for s in samples if s.group_id in val_groups]
    test = [s for s in samples if s.group_id in test_groups]
    return train, val, test
