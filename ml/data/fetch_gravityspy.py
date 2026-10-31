"""Download the labeled Gravity Spy glitch dataset.

Pulls the official Gravity Spy training set from its Zenodo release
(Bahaadini et al., record 1476156 -- "Machine learning for Gravity Spy:
Glitch classification and dataset") and extracts it into
data/raw/gravityspy/<label>/, one directory per glitch morphology class,
matching the layout ml/classifier/dataset.py expects.

The release ships a single HDF5 file (trainingsetv1d0.h5, ~3.3 GB) laid
out as /<label>/<sample_type>/<sample_id>/<duration>.png, where
sample_type is Gravity Spy's own train/validation/test split and each
leaf is a decoded (1, 140, 170) grayscale pixel array (not raw PNG
bytes). We ignore sample_type -- ml/classifier/dataset.py does its own
split, by event/glitch id -- and keep one fixed duration per sample so
each glitch contributes a single image, the same as one real-signal
spectrogram per confirmed event.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import h5py
import numpy as np

ZENODO_FILES_URL = "https://zenodo.org/records/1476156/files"
TRAINING_SET_H5 = "trainingsetv1d0.h5"


def _download_with_progress(url: str, out_path: Path) -> None:
    def report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = done * 100 // total_size
        print(f"\r{out_path.name}: {pct}% ({done // (1024 * 1024)} MB / {total_size // (1024 * 1024)} MB)",
              end="", flush=True)

    urllib.request.urlretrieve(url, out_path, reporthook=report)
    print()


def _download_training_set(archive_dir: Path) -> Path:
    archive_path = archive_dir / TRAINING_SET_H5
    if not archive_path.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        _download_with_progress(f"{ZENODO_FILES_URL}/{TRAINING_SET_H5}?download=1", archive_path)
    return archive_path


def _save_png(pixels: np.ndarray, out_path: Path) -> None:
    """Saves a (1, H, W) or (H, W) pixel array as an 8-bit grayscale PNG,
    min-max normalized so it displays regardless of the source value range.
    """
    import matplotlib.pyplot as plt

    image = np.squeeze(pixels)
    lo, hi = image.min(), image.max()
    normalized = (image - lo) / (hi - lo + 1e-12)
    plt.imsave(out_path, normalized, cmap="gray", vmin=0.0, vmax=1.0)


def fetch_gravityspy_dataset(
    dest_dir: str = "data/raw/gravityspy",
    duration: str = "1.0",
) -> dict[str, int]:
    """Downloads the Gravity Spy training set (skipped if already present)
    and extracts one PNG per labeled sample into `dest_dir`, one
    subdirectory per glitch class: `dest_dir/<label>/<sample_id>.png`.

    `duration` selects which of Gravity Spy's four spectrogram durations
    (0.5/1.0/2.0/4.0 seconds) to keep per sample.

    Returns {label: image_count}.
    """
    dest = Path(dest_dir)
    archive_path = _download_training_set(dest / "_archive")

    duration_key = f"{duration}.png"
    counts: dict[str, int] = {}

    with h5py.File(archive_path, "r") as f:
        for label in f.keys():
            out_label_dir = dest / label
            out_label_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            label_group = f[label]
            for sample_type in label_group.keys():
                sample_type_group = label_group[sample_type]
                for sample_id in sample_type_group.keys():
                    sample_group = sample_type_group[sample_id]
                    if duration_key not in sample_group:
                        continue
                    pixels = sample_group[duration_key][()]
                    _save_png(pixels, out_label_dir / f"{sample_id}.png")
                    count += 1

            counts[label] = count
            print(f"{label}: {count} images")

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest-dir", default="data/raw/gravityspy")
    parser.add_argument("--duration", default="1.0", choices=["0.5", "1.0", "2.0", "4.0"])
    args = parser.parse_args()

    counts = fetch_gravityspy_dataset(args.dest_dir, args.duration)
    total = sum(counts.values())
    print(f"Saved {total} images across {len(counts)} classes to {args.dest_dir}")
