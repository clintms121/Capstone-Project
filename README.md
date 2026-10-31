# Black Hole Merger Classifier

A capstone machine learning project. A neural network looks at real data
from the LIGO gravitational-wave detectors and tells black hole mergers
apart from detector glitches. A desktop app explains the science, shows the
model's results, and replays each merger in 3D.

## Start the program (Windows)

**One-time setup**

1. Install [MSYS2](https://www.msys2.org) (keep the default folder,
   `C:\msys64`). Open the **MSYS2 UCRT64** terminal from the Start menu and run:
   ```
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
   ```
2. Install [CMake](https://cmake.org/download/) and [Git](https://git-scm.com).

**Build and run.** In PowerShell, from the project folder:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
.\build\capstone.exe
```

The first build downloads its libraries, so it needs internet and takes a
few minutes. After that, you can start the app by double-clicking
`build\capstone.exe`.

## Using the app

- **Begin → View Research → Under the Hood:** slides on black holes, LIGO's
  data, spacetime, and how the model works, plus the Space Time Simulation.
- **Begin → Classifier:** pick a real LIGO merger to see the image the model
  looked at and how sure it is. **Watch both of these black holes merge**
  opens the 3D simulation.

## Rebuilding the data (optional)

The app shows results computed ahead of time by the Python code and saved in
`data/demo/` and `data/trajectories/`. You only need this to recompute them:
it downloads about 3.6 GB of data and trains the model (a GPU helps).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m ml.data.fetch_gravityspy
python -m ml.data.fetch_all_events
python -m ml.classifier.train --epochs 30
python -m ml.classifier.injection_test
python -m ml.mass_estimation.regression
python -m physics.build_trajectories
python -m ml.export_demo
```

See `ml/README.md` for what each step does and the results.

## What's where

- `graphics/` — the desktop app (C++). Each screen is its own `*_screen.cpp`.
- `ml/` — downloading LIGO data, turning it into images, the classifier, and
  the mass estimate (Python). Details in `ml/README.md`.
- `physics/` — computes each merger's orbit for the simulation (Python).
  Details in `physics/README.md`.
- `data/demo/`, `data/trajectories/` — the results the app shows.
- `imgs/`, `fonts/` — images and the font the app uses.
