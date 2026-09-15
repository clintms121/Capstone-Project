"""Whiten strain and produce fixed-size Q-transform spectrograms.

Stub. Both LIGO strain segments and Gravity Spy glitches must go through
the same transform so they share input shape for ml/classifier/model.py.
"""

def whiten_and_qtransform(strain, sample_rate: int, out_size=(128, 128)):
    raise NotImplementedError
