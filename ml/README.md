# ml — GW signal/glitch classifier + mass estimation

## Pipeline

1. **Data** (`ml/data/`)
   - `fetch_ligo_strain.py` — pulls open strain around a confirmed event
     straight from GWOSC's public JSON event API (`gwosc.org/eventapi/...`)
     over plain HTTP, not the `gwosc`/`gwpy` packages (see Environment below
     for why). `python -m ml.data.fetch_ligo_strain GW150914 --detector H1`
     downloads the hdf5 file to `data/raw/strain/` and prints its GPS time.
   - `fetch_gravityspy.py` — pull the labeled Gravity Spy glitch dataset
     (~20 morphological classes: Blip, Koi Fish, Scattered Light, ... plus
     confirmed real-signal examples) via the Gravity Spy Zenodo release.
     Still a stub.
   - Raw downloads land in `data/raw/` (gitignored — do not commit strain
     files or the Gravity Spy image set).

2. **Preprocessing** (`ml/preprocessing/spectrogram.py`)
   - Implemented. Loads a GWOSC hdf5 file directly with `h5py`, whitens
     against a Welch PSD estimate, band-passes to 20-500 Hz, and produces a
     fixed-size (128x128 by default) log-power spectrogram normalized to
     [0, 1] — the tensor a CNN would train on. Run directly for a sanity
     check: `python -m ml.preprocessing.spectrogram --input <hdf5> --event-gps <gps>`
     saves both the `.npy` array and a `.png` you can look at.
   - This is an STFT-based spectrogram, not gwpy's true Q-transform (same
     compiler blocker as above) — confirmed to show a visible chirp on
     GW150914 and GW170817; revisit with a real Q-transform once gwpy can
     be installed. Glitches from Gravity Spy need to go through this same
     function so both classes share input shape.

3. **Classifier** (`ml/classifier/`)
   - `dataset.py` — loads spectrogram tensors + labels (glitch classes plus
     a "real signal" class), train/val/test split by event/glitch id (never
     split within the same event).
   - `model.py` — CNN (start with a small ResNet-style stack; spectrograms
     are single-channel images).
   - `train.py` — training loop, checkpointing, confusion matrix over all
     ~20+1 classes (glitch classes are imbalanced — use class weighting).

4. **Mass estimation** (`ml/mass_estimation/`)
   - Only run on inputs the classifier calls "real signal".
   - `regression.py` — regress chirp mass and mass ratio from the
     time-frequency track (the chirp's frequency-vs-time slope encodes
     chirp mass directly via the PN relation
     `f_GW(t) ~ (5/(256*(tc-t)))^(3/8) * (G*Mchirp/c^3)^(-5/8)`).
   - Sanity-check against the published GWTC parameters for confirmed
     events — this is a regression model, not a full Bayesian PE pipeline
     (that's out of scope for this project; note the simplification in the
     writeup).
   - Output: `(chirp_mass, mass_ratio)` per event, written to
     `data/mass_estimates.csv`, consumed by `physics/inspiral.py`.

## Environment

See `requirements.txt` at the repo root (`.venv`, Python 3.12 — 3.12 chosen
over the machine's default 3.14 for wider package/wheel compatibility).
`gwpy`/`gwosc` are listed but not currently installable here: they pull in
`igwn-segments`, which needs an MSVC C compiler to build and this machine's
Visual Studio 2022 install is missing the C++ workload. `fetch_ligo_strain.py`
and `spectrogram.py` work around this with plain HTTP + `h5py` instead.
Install `gwpy`/`gwosc` once "Desktop development with C++" is added to the
VS install, and switch those two scripts back to gwpy's fetch/whiten/
q_transform if a true Q-transform is worth the dependency at that point.

## Status

- `fetch_ligo_strain.py` — implemented, tested against GW150914, GW170817,
  GW170823.
- `preprocessing/spectrogram.py` — implemented, verified against the same
  three events (clean chirp on GW150914's high-SNR BBH merger, correctly
  slow/gradual chirp on GW170817's neutron-star merger, noisier result on
  the lower-SNR GW170823 — matches expectation in all three cases).
- `fetch_gravityspy.py`, `classifier/`, `mass_estimation/` — still stubs.
  Next up: `fetch_gravityspy.py`, since the classifier needs both classes
  of data before it can be built.
