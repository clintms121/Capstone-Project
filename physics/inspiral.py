"""Post-Newtonian binary inspiral trajectory generator.

Leading-order (quadrupole) orbital decay of a circular binary, in geometric
units (G = c = 1) with lengths and times in units of the total mass M:

    da/dt = -64/5 * eta / a^3      =>  a(t)^4 = a0^4 - (256/5) * eta * t
    dphi/dt = a^(-3/2)             =>  phi(a) = (a0^(5/2) - a^(5/2)) / (32 * eta)

Both solved analytically. The orbit is followed past the ISCO (a = 6 M),
where post-Newtonian theory no longer applies, down to a = 2 M, where the two
Schwarzschild horizons (radius 2 m_i) touch -- a visual plunge, not physics;
ringdown.py takes over from there. In these units the trajectory depends
only on the mass ratio; the total mass just sets the time scale (seconds per
M = M * G * M_sun / c^3).
"""

from __future__ import annotations

import numpy as np

T_SUN = 4.925491025543576e-6  # G * M_sun / c^3, seconds


def component_masses(chirp_mass_msun: float, mass_ratio: float) -> tuple[float, float]:
    """(m1, m2) with m2 = q * m1 from chirp mass and mass ratio q <= 1."""
    q = min(mass_ratio, 1 / mass_ratio)
    m1 = chirp_mass_msun * (1 + q) ** 0.2 * q ** -0.6
    return m1, q * m1


def generate_inspiral(
    m1_msun: float,
    m2_msun: float,
    a0_over_m: float = 15.0,
    merge_sep_over_m: float = 2.0,
    samples_per_orbit: int = 60,
) -> dict[str, np.ndarray | float]:
    """Inspiral + plunge from separation `a0_over_m` to `merge_sep_over_m`,
    sampled uniformly in orbital phase so the fast final orbits are as smooth
    as the slow early ones (time steps shrink toward merger). Returns t (units
    of M), separation a, orbital phase phi, body positions (n, 3) for both
    components (center of mass at the origin, orbit in the x-y plane), a
    quadrupole-order plus-polarization strain h (unnormalized), and scalars
    eta, total mass, seconds per M.
    """
    m_total = m1_msun + m2_msun
    eta = m1_msun * m2_msun / m_total**2
    phi_merge = (a0_over_m**2.5 - merge_sep_over_m**2.5) / (32 * eta)

    phi = np.linspace(0.0, phi_merge, int(np.ceil(phi_merge / (2 * np.pi) * samples_per_orbit)) + 1)
    a = np.maximum(a0_over_m**2.5 - 32 * eta * phi, merge_sep_over_m**2.5) ** 0.4
    t = np.maximum(5 * (a0_over_m**4 - a**4) / (256 * eta), 0.0)

    orbit = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
    pos1 = orbit * (a * m2_msun / m_total)[:, None]
    pos2 = -orbit * (a * m1_msun / m_total)[:, None]

    return {
        "t": t,
        "a": a,
        "phi": phi,
        "pos1": pos1,
        "pos2": pos2,
        "h": eta / a * np.cos(2 * phi),
        "eta": eta,
        "m_total_msun": m_total,
        "seconds_per_m": m_total * T_SUN,
    }
