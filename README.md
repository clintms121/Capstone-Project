## Capstone Project

Gravitational wave classification + merger visualization.

A CNN trained on LIGO open strain data (GWOSC) and the labeled Gravity Spy
glitch dataset separates real compact-binary signals from detector
artifacts. Confirmed signals go through a chirp-mass/mass-ratio regression,
and those parameters drive a 3D post-Newtonian inspiral + ringdown
animation, rendered in a GLFW/ImGui/OpenGL app.

## Layout

- `graphics/` — the renderer (title -> menu -> research/simulation screens)
- `ml/` — data fetch, spectrogram preprocessing, classifier, mass estimation
  (see `ml/README.md`)
- `physics/` — inspiral/ringdown trajectory model consumed by the renderer
  (see `physics/README.md`)
- `models/` — downloaded HST imaging of black-hole host galaxies from an
  earlier project direction; not used by the current pipeline

## Build (C++ renderer)

```
cmake -S . -B build
cmake --build build
```

## Python pipeline

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
