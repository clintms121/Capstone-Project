# physics — binary merger trajectory model

Supplies the trajectory that drives the 3D visualization in
`graphics/simulation_screen.cpp`. Given a chirp mass and mass ratio (from
`ml/mass_estimation`), this generates a full inspiral -> merger -> ringdown
track that the renderer just plays back — no physics runs inside the render
loop.

## Stages

1. **Inspiral** (`inspiral.py`): post-Newtonian (leading-order, quadrupole)
   two-body orbit. Orbital separation shrinks via GW energy loss:
   `da/dt = -64/5 * G^3 * m1 * m2 * (m1+m2) / (c^5 * a^3)`.
   Integrate `a(t)` and orbital phase `phi(t)` until separation reaches the
   effective ISCO (`a ~ 6 GM/c^2` for the total mass).
2. **Merger**: short transition segment blending the end of the PN inspiral
   into the ringdown's initial amplitude/frequency (no full numerical
   relativity — this is a visualization, not a waveform model).
3. **Ringdown** (`ringdown.py`): damped sinusoid at the remnant's fundamental
   quasinormal mode frequency/damping time (Kerr QNM fit, e.g. Berti et al.
   fitting formulas), remnant mass/spin from NR fitting formulas
   (e.g. Healy & Lousto).

## Output

A time series of `(t, x1, y1, z1, x2, y2, z2)` (or a single separation +
phase track, since the orbit is planar) sampled at a fixed frame rate,
written to `data/trajectories/<event_id>.npz`. No physics dependency needed
on the C++ side — it only plays the file back.

Every time a new event is processed, its row must also be appended to
`data/trajectories/index.csv` (create it with the header below if it
doesn't exist yet) — this is what `graphics/manifest.cpp` reads to populate
the event picker on the Simulation screen:

```
event_id,chirp_mass_msun,mass_ratio,path
GW150914,28.6,0.82,data/trajectories/GW150914.npz
```

## Status

Stub — not yet implemented. Depends on `ml/mass_estimation` producing
`(chirp_mass, mass_ratio)` per classified event first.
