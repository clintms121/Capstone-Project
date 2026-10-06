"""Download open strain data around confirmed GWTC events from GWOSC.

Talks to GWOSC's public JSON event API directly over HTTP rather than the
gwosc/gwpy packages, which need a compiler (see requirements.txt) not
available on this machine. Saves files to data/raw/strain/, in the layout
ml/preprocessing/spectrogram.py expects.

GWOSC only publishes short (32 s) files for some events and catalog versions
-- e.g. GW150914's latest (GWTC-2.1) entry lists only 4096 s files, while its
GWTC-1 entry still has 32 s ones, and O4 events have 4096 s files only
(~110 MB each). So every catalog version is checked for a short file first;
failing that, just the needed window is range-read out of the long file over
HTTP and saved locally in the same GWOSC layout.
"""

from __future__ import annotations

import functools
import io
import json
import re
import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi
import h5py

ALL_EVENTS_URL = "https://gwosc.org/eventapi/json/allevents/"

# gwosc.org chains to an emSign root that Windows only fetches on demand, which
# Python's ssl module never triggers -- use Mozilla's CA bundle instead.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


@functools.lru_cache(maxsize=None)
def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, context=_SSL_CONTEXT) as resp:
        return json.loads(resp.read())


def _download(url: str, out_path: Path) -> None:
    with urllib.request.urlopen(url, context=_SSL_CONTEXT) as resp, open(out_path, "wb") as f:
        shutil.copyfileobj(resp, f)


class _HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file via HTTP Range requests, so
    h5py can read one dataset slice without downloading the whole file.
    """

    BLOCK = 512 * 1024

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.blocks: dict[int, bytes] = {}
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, context=_SSL_CONTEXT) as resp:
            self.size = int(resp.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, index: int) -> bytes:
        if index not in self.blocks:
            start = index * self.BLOCK
            end = min(start + self.BLOCK, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, context=_SSL_CONTEXT) as resp:
                self.blocks[index] = resp.read()
        return self.blocks[index]

    def readinto(self, buffer) -> int:
        n = min(len(buffer), max(0, self.size - self.pos))
        out = memoryview(buffer)
        done = 0
        while done < n:
            index, offset = divmod(self.pos + done, self.BLOCK)
            chunk = self._block(index)[offset:offset + n - done]
            out[done:done + len(chunk)] = chunk
            done += len(chunk)
        self.pos += n
        return n


def _crop_remote_strain(url: str, crop_start: int, duration_s: int, out_path: Path) -> None:
    with h5py.File(_HttpRangeFile(url), "r") as src:
        dset = src["strain/Strain"]
        dt = float(dset.attrs["Xspacing"])
        x0 = float(dset.attrs["Xstart"])
        i0 = int(round((crop_start - x0) / dt))
        n = int(round(duration_s / dt))
        if i0 < 0 or i0 + n > dset.shape[0]:
            raise ValueError(f"{duration_s}s window at GPS {crop_start} falls outside {Path(url).name}")
        data = dset[i0:i0 + n]

    with h5py.File(out_path, "w") as dst:
        out = dst.create_dataset("strain/Strain", data=data)
        out.attrs["Xspacing"] = dt
        out.attrs["Xstart"] = x0 + i0 * dt


def list_confident_events() -> list[dict]:
    """Returns one entry per GWOSC event name (latest catalog version only),
    each with at least commonName, GPS, catalog.shortName, and jsonurl.

    Despite the name, this includes marginal candidates and non-LVK catalogs
    -- see ml/data/fetch_all_events.py:select_events for confidence filtering.
    """
    latest: dict[str, dict] = {}
    for info in _get_json(ALL_EVENTS_URL)["events"].values():
        name = info["commonName"]
        if name not in latest or info["version"] > latest[name]["version"]:
            latest[name] = info
    return list(latest.values())


def _strain_entries_all_versions(event_name: str) -> list[dict]:
    """Every strain entry listed for `event_name`, across all of its catalog
    versions, newest version first.
    """
    versions = sorted(
        (info for info in _get_json(ALL_EVENTS_URL)["events"].values() if info["commonName"] == event_name),
        key=lambda info: info["version"],
        reverse=True,
    )
    entries = []
    for info in versions:
        event_data = _get_json(info["jsonurl"])
        key = next(iter(event_data["events"]))
        entries += event_data["events"][key]["strain"]
    return entries


def fetch_event_strain(
    event_name: str,
    detector: str = "H1",
    sample_rate_hz: int = 4096,
    duration_s: int = 32,
    out_dir: str = "data/raw/strain",
) -> tuple[Path, float]:
    """Gets a `duration_s` strain file (hdf5) around `event_name` for
    `detector` and returns (local_path, event_gps_time). Uses a published
    file of that length if any catalog version has one, otherwise range-reads
    the window out of a longer file. Skips the fetch if already present.
    """
    events = {e["commonName"]: e for e in list_confident_events()}
    if event_name not in events:
        raise ValueError(f"No event named {event_name!r} in the GWOSC catalog")
    gps = float(events[event_name]["GPS"])

    candidates = [
        e for e in _strain_entries_all_versions(event_name)
        if e["format"] == "hdf5" and e["detector"] == detector and e["sampling_rate"] == sample_rate_hz
    ]
    if not candidates:
        raise ValueError(f"No hdf5 strain for {event_name}/{detector} at {sample_rate_hz} Hz")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    exact = next((e for e in candidates if e["duration"] == duration_s), None)
    if exact is not None:
        out_path = out / Path(exact["url"]).name
        if not out_path.exists():
            _download(exact["url"], out_path)
        return out_path, gps

    crop_start = int(gps) - duration_s // 2
    source = next(
        (e for e in candidates
         if e["GPSstart"] <= crop_start and crop_start + duration_s <= e["GPSstart"] + e["duration"]),
        None,
    )
    if source is None:
        raise ValueError(f"No {detector} strain file covers the {duration_s}s window around {event_name}")

    stem = re.sub(r"-\d+-\d+$", f"-{crop_start}-{duration_s}", Path(source["url"]).stem)
    out_path = out / f"{stem}.hdf5"
    if not out_path.exists():
        _crop_remote_strain(source["url"], crop_start, duration_s, out_path)
    return out_path, gps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", help="e.g. GW150914")
    parser.add_argument("--detector", default="H1")
    parser.add_argument("--sample-rate", type=int, default=4096)
    parser.add_argument("--duration", type=int, default=32)
    parser.add_argument("--out-dir", default="data/raw/strain")
    args = parser.parse_args()

    path, gps = fetch_event_strain(
        args.event, args.detector, args.sample_rate, args.duration, args.out_dir
    )
    print(f"Saved {path} (event GPS {gps})")
