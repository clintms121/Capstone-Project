# physics — binary merger trajectory model

Supplies the trajectory that drives the 3D visualization in
`graphics/simulation_screen.cpp`. Given a chirp mass and mass ratio (from
`ml/mass_estimation`), this generates a full inspiral -> merger -> ringdown
track that the renderer just plays back — no physics runs inside the render
loop.

```
python -m physics.build_trajectories
```
reads `data/mass_estimates.csv`, writes `data/trajectories/<event_id>.csv`
per event, and rewrites `data/trajectories/index.csv`.

## Stages

Everything is in geometric units (G = c = 1) with lengths and times in units
of the total mass M. In those units a non-spinning binary's trajectory
depends only on the mass ratio; the total mass only sets the time scale
(`t_s` column = `t_m` × M × 4.93 µs per solar mass).

1. **Inspiral** (`inspiral.py`): leading-order (quadrupole) post-Newtonian
   circular orbit, solved analytically —
   `a(t)^4 = a0^4 - (256/5) η t`, `φ(a) = (a0^(5/2) - a^(5/2)) / (32 η)` —
   from a0 = 15 M (~17 orbits for equal masses).
2. **Merger**: the same equations are followed past the ISCO (6 M), where
   post-Newtonian theory stops being valid, down to a = 2 M where the two
   Schwarzschild horizons touch. A visual plunge, not a waveform model.
3. **Ringdown** (`ringdown.py`): remnant spin and mass from non-spinning
   numerical-relativity fits (spin: Rezzolla et al. 2008), ringing in the
   l = m = 2, n = 0 Kerr quasinormal mode (Berti, Cardoso & Will 2006 fits).
   Checks against GW150914: remnant spin 0.687 (measured 0.69), ringdown
   frequency 253 Hz (observed ~250 Hz), ~4.8% of the mass radiated
   (measured ~4.7%).

## Trajectory file contract

`data/trajectories/<event_id>.csv`, header row then one row per sample:

| column | meaning |
|---|---|
| `t_m`, `t_s` | time in units of M, and in seconds |
| `x1 y1 z1`, `x2 y2 z2` | body positions (units of M, center of mass at origin, orbit in x-y plane) |
| `r1`, `r2` | horizon radii: 2 m_i / M before merger; the remnant's Kerr r₊ after (both bodies then sit at the origin) |
| `h` | plus-polarization strain, normalized to peak \|h\| = 1 |
| `merged` | 0 inspiral, 1 ringdown |

Samples are uniform in orbital phase (60 per orbit) during the inspiral and
30 per quasinormal cycle during ringdown — **not** uniform in time, so the
fast final orbits are as smooth as the early ones. Play back by
interpolating on `t_m` (or `t_s`). CSV rather than `.npz` so the C++ side can
read it with a plain `ifstream`, like `graphics/manifest.cpp` reads the index.

`data/trajectories/index.csv` (read by `graphics/manifest.cpp`):

```
event_id,chirp_mass_msun,mass_ratio,path
GW150914,30.690,0.867,data/trajectories/GW150914.csv
```

Chirp masses here are detector-frame (see `ml/README.md`).

## Status

Implemented. The C++ playback of these files (simulation_screen.cpp) is not
written yet — the screen currently shows the selected event's parameters
and trajectory path only.
