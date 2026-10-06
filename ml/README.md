# ml — GW signal/glitch classifier + mass estimation

## Pipeline

1. **Data** (`ml/data/`)
   - `fetch_ligo_strain.py` — gets a 32 s strain file around one event
     straight from GWOSC's public JSON event API over plain HTTP, not the
     `gwosc`/`gwpy` packages (see Environment below for why).
     `python -m ml.data.fetch_ligo_strain GW150914 --detector H1` saves to
     `data/raw/strain/`. Checks every catalog version for a published 32 s
     file (newer versions of O1/O2 events dropped them); for O4 events, which
     only have ~110 MB 4096 s files, it range-reads just the 32 s window over
     HTTP (~1 MB) and saves it in the same GWOSC layout.
   - `fetch_all_events.py` — batch version: every LVK-confident event with
     network SNR >= 10 (194 events), H1 + L1, each turned into a Q-scan in
     `data/processed/real_signal/`, plus `real_signal_manifest.csv` with the
     published chirp mass / mass ratio per spectrogram (regression targets
     for step 4), plus off-source noise Q-scans in `data/processed/noise/`
     (see the shortcut-learning note under Classifier). Marginal candidates,
     the third-party IAS catalog, and
     quieter events are excluded so nothing noise-like gets the real_signal
     label. `python -m ml.data.fetch_all_events`
   - `fetch_gravityspy.py` — downloads the Gravity Spy training set from
     Zenodo (record 1476156, `trainingsetv1d0.h5`, ~3.3 GB) and extracts one
     1.0 s image per labeled glitch into `data/raw/gravityspy/<label>/`.
     `python -m ml.data.fetch_gravityspy`
   - Everything lands under `data/` (gitignored).

2. **Preprocessing** (`ml/preprocessing/spectrogram.py`)
   - `make_qscan` — the classifier input. A numpy constant-Q transform (same
     algorithm as gwpy's `q_transform`) laid out exactly like Gravity Spy's
     images: 140x170, log 10-2048 Hz with high frequency at the top, 1 s
     centered on the event, normalized energy clipped to [0, 25.5]. Matching
     Gravity Spy's layout matters: with the earlier STFT spectrograms, the
     two classes differed in orientation, frequency axis and background
     texture, so a CNN could separate them without learning anything about
     chirps.
   - `make_spectrogram` — the original STFT (20-500 Hz, 128x128) version,
     kept for the CLI sanity-check plots; not used for training.

3. **Classifier** (`ml/classifier/`)
   - `dataset.py` — loads both sources with identical per-image scaling,
     merges Gravity Spy's `Chirp` class (hardware-injected binary signals)
     into `real_signal`, and splits train/val/test by group id (same event
     across detectors, or same glitch id), stratified per class.
   - **Shortcut-learning check (keep this in the writeup):** the first full
     training run scored a perfect 65/65 real signals with 0 false alarms --
     and then called 115/115 *pure-noise* images real_signal once that noise
     was run through our Q-scan. Every image our pipeline produced had been a
     real signal, so the model learned "our pipeline -> real". Fix:
     `fetch_all_events.py` also cuts off-source noise from each strain file
     (±6 s and ±12 s from the event; post-merger only for chirp mass < 5,
     whose inspiral is still in band before merger) into
     `data/processed/noise/`, labeled `No_Glitch` and grouped with its event.
     `train.py`'s report now breaks down real_signal calls by image source
     so this is checked on every run.
   - Results after the fix (held-out test split, 1,559 images, 22 classes):
     97.9% accuracy, macro-F1 0.96; real_signal recall 0.985, precision
     0.89; our-pipeline noise called real 8/216, Gravity Spy noise 0/27.
   - `injection_test.py` — the number to quote. Simulated TaylorF2 chirps
     (Mc_det 8-35) injected into held-out detector noise at known SNR,
     rendered with the same Q-scan: caught 5% at SNR 4, 46% at 8, 64% at
     10, 89% at 15, 93% at 20+; plain noise at the same times flagged 1.8%.
     A smooth efficiency curve rising from the false-alarm floor is what a
     classifier that actually sees chirps should produce. Caveat: noise
     within ~3 s of a loud event is flagged much more often (42/108) than
     noise 9 s away (1/108) -- event energy reaches neighboring windows
     through long low-frequency Q tiles -- so the classifier is only
     meaningful on windows centered on a trigger.
     `python -m ml.classifier.injection_test`
   - `model.py` — small ResNet-style CNN (~2.8M params), single-channel
     input, global average pooling.
   - `train.py` — class-weighted cross-entropy, random time-shift
     augmentation, best checkpoint by val macro-F1 to
     `data/checkpoints/classifier_best.pt`; test report and confusion matrix
     to `data/reports/`. `python -m ml.classifier.train --epochs 30`

4. **Mass estimation** (`ml/mass_estimation/regression.py`)
   - Matched filtering, the same idea LIGO's searches use, simplified:
     non-spinning TaylorF2 templates (2PN phase, inspiral only, cut at ISCO
     and at 1 kHz), one detector at a time, best template by chi-square
     reweighted SNR. The bank has 1,540 chirp masses spaced ~Mc^(5/3) in log
     (a neutron-star merger's SNR peak is ~0.1% wide in Mc, a heavy BBH's
     tens of percent) x a mass-ratio refinement. Loud glitches are gated
     (zeroed with a taper) in the *whitened* data first.
   - Why not regress from the Q-scan image: a prototype extracting the
     chirp track from the 140x170 image barely beat predicting the median
     (39% vs. 46% median error) -- the image is too coarse. Matched filtering
     uses the full phase of the strain.
   - Measures *detector-frame* chirp mass (1+z)*Mc_source -- one waveform
     can't separate mass from redshift -- so it's scored against published
     detector-frame values.
   - **Noise threshold, measured not tuned:** with ~3,000 templates, pure
     noise always matches something at SNR 5-6, with an essentially random
     winner. The same search on 40 signal-free windows (9 s after events)
     peaks at SNR 6.4, so only estimates above that count. Results (178
     events): above threshold, 76 events, **median error 2.2%, 91% within
     10%** (1.3% for Mc_det < 15, 2.7% for 15-30, 7% for 30-60); below it,
     78% median error. Only 2/21 heavy systems (Mc_det >= 60) clear it --
     their SNR is in the merger-ringdown inspiral-only templates lack. A
     log-linear calibration was tried and removed: noise-dominated outliers
     made it worse than the raw estimate.
   - Mass ratio only enters the phase at 1PN and is never better than the
     population median (MAE 0.25 vs. 0.11), so outputs use the median (0.70)
     and say so. With q fixed, every trajectory has the same shape in units
     of total mass; chirp mass sets each event's time scale.
   - Events the classifier calls real_signal (on at least one detector)
     *and* above the threshold go to `data/mass_estimates.csv`, consumed by
     `physics/build_trajectories.py`. Report + scatter plot in
     `data/reports/`. Results are cached in
     `data/processed/real_signal/matched_filter*.csv` (~13 s per detector).
     `python -m ml.mass_estimation.regression`
   - Point estimates, not Bayesian parameter estimation -- note in writeup.

## Environment

See `requirements.txt` at the repo root (`.venv`, Python 3.12 — 3.12 chosen
over the machine's default 3.14 for wider package/wheel compatibility).
PyTorch is the CUDA 12.8 build (`pip install torch --index-url
https://download.pytorch.org/whl/cu128`) so training runs on the RTX 4080.
`certifi` is required: gwosc.org's certificate chains to a root Windows only
fetches on demand, which Python's `ssl` never triggers.

`gwpy`/`gwosc` are listed but not currently installable here: they pull in
`igwn-segments`, which needs an MSVC C compiler to build and this machine's
Visual Studio 2022 install is missing the C++ workload. The fetch and
Q-transform code above reimplements what was needed with plain HTTP, `h5py`
and numpy instead.

## Known caveats (note in the writeup)

- Whitening uses a PSD estimated from the same 32 s that contains the event,
  which slightly biases it; a production pipeline uses separate off-source
  data.
- GW170817's L1 data contains a famous loud glitch ~1 s before merger; its
  Q-scan is dominated by it but still carries the real_signal label.
- Real signals are still a small class (345 Q-scans from 192 events + 66
  merged Chirp vs. ~8,500 glitches); class weighting compensates, but
  per-class metrics matter more than overall accuracy.
- GWOSC's own detector-frame `chirp_mass` field is empty for every event;
  the manifest computes it as `chirp_mass_source * (1 + redshift)`.

## Status

- Data, preprocessing, classifier, injection test, and mass estimation:
  implemented and run end to end. `physics/build_trajectories.py` has
  turned the 76 measurable events into `data/trajectories/` (the
  `(TEST DATA)` rows in `index.csv` are gone).
- `export_demo.py` writes `data/demo/` for the app's Classifier Demo screen:
  every real-merger recording plus held-out glitches and noise, each with
  its Q-scan (PGM), the classifier's top-3 guesses, the matched-filter mass,
  and whether it was held out; plus the test-split metrics and injection
  curve. The app replays these results; it doesn't run the models.
- Next: C++ playback of the trajectory CSVs in
  `graphics/simulation_screen.cpp` (see `physics/README.md` for the format).

Full pipeline, in order:
```
python -m ml.data.fetch_gravityspy
python -m ml.data.fetch_all_events
python -m ml.classifier.train --epochs 30
python -m ml.classifier.injection_test
python -m ml.mass_estimation.regression
python -m physics.build_trajectories
python -m ml.export_demo
```
