"""Post-Newtonian binary inspiral trajectory generator.

Stub. Will integrate the leading-order quadrupole orbital-decay equation
da/dt = -64/5 * G^3 * m1 * m2 * (m1 + m2) / (c^5 * a^3)
from an initial separation down to the effective ISCO, tracking orbital
phase alongside it, and hand off to ringdown.py at merger.
"""

def generate_inspiral(m1_msun: float, m2_msun: float, a0_over_risco: float = 20.0):
    raise NotImplementedError
