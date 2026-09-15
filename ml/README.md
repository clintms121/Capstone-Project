# ml — GW signal/glitch classifier + mass estimation

## Pipeline

1. **Data** (`ml/data/`)
   - `fetch_ligo_strain.py` — pull open strain around confirmed events from
     GWOSC via the `gwosc`/`gwpy` packages (e.g. the GWTC catalog event list).
   - `fetch_gravityspy.py` — pull the labeled Gravity Spy glitch dataset
     (~20 morphological classes: Blip, Koi Fish, Scattered Light, ... plus
     confirmed real-signal examples) via the Gravity Spy Zenodo release.
   - Raw downloads land in `data/raw/` (gitignored — do not commit strain
     files or the Gravity Spy image set).

2. **Preprocessing** (`ml/preprocessing/spectrogram.py`)
   - Whiten strain against the detector PSD, band-pass, then Q-transform to
     a fixed-size time-frequency image. Same transform for LIGO events and
     Gravity Spy glitches so both feed the same classifier input shape.

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

See `requirements.txt` at the repo root. Needs `gwpy`, `gwosc`, `torch`,
`numpy`, `scipy`, `pandas`, `scikit-learn`.

## Status

Stub — no code implemented yet. Start with `ml/data/fetch_gravityspy.py`
and `ml/preprocessing/spectrogram.py` since every later stage depends on
having spectrograms in hand.
