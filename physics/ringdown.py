"""Post-merger ringdown: remnant properties and quasinormal-mode settling.

Remnant from fits to non-spinning numerical-relativity mergers, as functions
of the symmetric mass ratio eta = m1 m2 / M^2 (units of the total mass M):

    spin    a_f   = 2 sqrt(3) eta - 3.871 eta^2 + 4.028 eta^3   (Rezzolla et al. 2008)
    mass    M_f/M = 1 + (sqrt(8/9) - 1) eta - 0.4333 eta^2 - 0.4392 eta^3

(equal masses: a_f ~ 0.69, ~4.8% of M radiated -- GW150914 measured 0.69 and
~3.1 of 65.3 M_sun, ~4.7%). The remnant rings down in the fundamental l = m = 2, n = 0 Kerr
quasinormal mode, with frequency and quality factor from Berti, Cardoso &
Will (2006):

    M_f * omega = 1.5251 - 1.1568 (1 - a_f)^0.1292
    Q           = 0.7000 + 1.4187 (1 - a_f)^-0.4990,   tau = 2 Q / omega
"""

from __future__ import annotations

import numpy as np


def remnant_mass_spin(eta: float) -> tuple[float, float]:
    """(M_f / M, dimensionless spin a_f) of the merged black hole."""
    spin = 2 * np.sqrt(3) * eta - 3.871 * eta**2 + 4.028 * eta**3
    mass = 1 + (np.sqrt(8 / 9) - 1) * eta - 0.4333 * eta**2 - 0.4392 * eta**3
    return float(mass), float(spin)


def qnm_220(remnant_mass: float, remnant_spin: float) -> tuple[float, float]:
    """(angular frequency omega, damping time tau) of the l=m=2, n=0 mode, in
    the same units as `remnant_mass` (units of M if remnant_mass = M_f / M).
    """
    omega = (1.5251 - 1.1568 * (1 - remnant_spin) ** 0.1292) / remnant_mass
    quality = 0.7000 + 1.4187 * (1 - remnant_spin) ** -0.4990
    return float(omega), float(2 * quality / omega)


def kerr_horizon_radius(remnant_mass: float, remnant_spin: float) -> float:
    """Outer horizon radius r_+ = M_f (1 + sqrt(1 - a_f^2))."""
    return remnant_mass * (1 + np.sqrt(1 - remnant_spin**2))


def generate_ringdown(
    remnant_mass: float,
    remnant_spin: float,
    amplitude: float,
    phase: float,
    samples_per_cycle: int = 30,
    n_damping_times: float = 10.0,
) -> dict[str, np.ndarray | float]:
    """Damped-sinusoid strain h(t) = A exp(-t/tau) cos(omega t + phase), for
    `n_damping_times` damping times starting at the merger (t = 0), in units
    of M.
    """
    omega, tau = qnm_220(remnant_mass, remnant_spin)
    t = np.arange(0.0, n_damping_times * tau, 2 * np.pi / omega / samples_per_cycle)
    return {
        "t": t,
        "h": amplitude * np.exp(-t / tau) * np.cos(omega * t + phase),
        "omega": omega,
        "tau": tau,
        "horizon_radius": kerr_horizon_radius(remnant_mass, remnant_spin),
    }
