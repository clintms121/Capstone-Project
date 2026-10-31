"""Turn data/mass_estimates.csv into trajectory files the renderer plays back.

For each classified event: component masses from (chirp mass, mass ratio),
PN inspiral + plunge (inspiral.py), then the remnant's quasinormal ringdown
(ringdown.py), written to data/trajectories/<event_id>.csv. Rewrites
data/trajectories/index.csv (read by graphics/manifest.cpp) with one row per
event.

Trajectory CSV columns, one row per sample. Samples are uniform in orbital
phase (60 per orbit) during the inspiral and 30 per quasinormal cycle in the
ringdown, NOT uniform in time -- play back by interpolating on t_m:
    t_m, t_s         time in units of total mass M, and in seconds
    x1 y1 z1         body 1 position (units of M, center of mass at origin)
    x2 y2 z2         body 2 position
    r1, r2           horizon radii (2 m_i before merger, Kerr r_+ of the
                     remnant after; both bodies sit at the origin then)
    h                plus-polarization strain, normalized to peak |h| = 1
    merged           0 during inspiral, 1 during ringdown
CSV rather than .npz so the C++ side can read it with a plain ifstream.

    python -m physics.build_trajectories
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from physics.inspiral import component_masses, generate_inspiral
from physics.ringdown import generate_ringdown, remnant_mass_spin

COLUMNS = ["t_m", "t_s", "x1", "y1", "z1", "x2", "y2", "z2", "r1", "r2", "h", "merged"]


def build_trajectory(chirp_mass_msun: float, mass_ratio: float) -> np.ndarray:
    m1, m2 = component_masses(chirp_mass_msun, mass_ratio)
    insp = generate_inspiral(m1, m2)
    m_total = insp["m_total_msun"]
    n_inspiral = len(insp["t"])

    remnant_mass, remnant_spin = remnant_mass_spin(insp["eta"])
    ring = generate_ringdown(
        remnant_mass, remnant_spin,
        amplitude=insp["eta"] / insp["a"][-1],
        phase=2 * insp["phi"][-1],
    )

    n_ring = len(ring["t"]) - 1  # first ringdown sample coincides with the merger frame
    t_m = np.concatenate([insp["t"], insp["t"][-1] + ring["t"][1:]])
    zeros = np.zeros((n_ring, 3))
    pos1 = np.concatenate([insp["pos1"], zeros])
    pos2 = np.concatenate([insp["pos2"], zeros])
    r1 = np.concatenate([np.full(n_inspiral, 2 * m1 / m_total), np.full(n_ring, ring["horizon_radius"])])
    r2 = np.concatenate([np.full(n_inspiral, 2 * m2 / m_total), np.full(n_ring, ring["horizon_radius"])])
    h = np.concatenate([insp["h"], ring["h"][1:]])
    merged = np.concatenate([np.zeros(n_inspiral), np.ones(n_ring)])

    return np.column_stack([
        t_m, t_m * insp["seconds_per_m"], pos1, pos2, r1, r2, h / np.abs(h).max(), merged,
    ])


def build_all(
    estimates_csv: str = "data/mass_estimates.csv",
    out_dir: str = "data/trajectories",
) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(estimates_csv, newline="") as f:
        events = list(csv.DictReader(f))

    index_rows = []
    for e in events:
        mc, q = float(e["chirp_mass_msun"]), float(e["mass_ratio"])
        path = out / f"{e['event_id']}.csv"
        np.savetxt(path, build_trajectory(mc, q), delimiter=",", header=",".join(COLUMNS),
                   comments="", fmt="%.6g")
        index_rows.append([e["event_id"], f"{mc:.3f}", f"{q:.3f}", path.as_posix()])

    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "chirp_mass_msun", "mass_ratio", "path"])
        writer.writerows(index_rows)
    return len(index_rows)


if __name__ == "__main__":
    n = build_all()
    print(f"Wrote {n} trajectories and data/trajectories/index.csv")
