"""Regress chirp mass and mass ratio from a classified real-signal spectrogram.

Stub. The chirp's frequency-vs-time slope encodes chirp mass directly via
f_GW(t) ~ (5 / (256 * (tc - t)))^(3/8) * (G * Mchirp / c^3)^(-5/8).
This is a point-estimate regression, not full Bayesian parameter estimation
— validate against published GWTC values for confirmed events and note the
simplification in the writeup.
"""

def estimate_masses(spectrogram):
    raise NotImplementedError
