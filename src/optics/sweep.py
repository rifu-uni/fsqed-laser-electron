"""``kappa_eff(f-number)`` sweep -> comparison with Fig. 4 of Olofsson & Gonoskov.

Fig. 4 plots ``chi_max`` as a function of the parabolic-mirror f-number for
``P = 1 PW``, ``eps = 1 GeV`` and ``lambda = 1 um``, with the peak electric/magnetic
field on the right axis.  The curve starts from the analytic Sheppard-Larkin
(bi-dipole) input and saturates at ``kappa ~ 5.25`` for ``f-number <= 0.25`` (the full
``-y`` hemisphere).

``chi_max`` scales as ``chi ~ eps * lambda^-1 * P^{1/2}``; ``a0`` (the field strength in
relativistic units) scales as ``P^{1/2}`` and is wavelength-independent for a fixed
f-number.
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from . import focus


def focal_field_vector(
    far_field,
    *,
    f_number: float,
    lambda0_m: float,
    power_w: float,
    ntheta: int = 256,
    nphi: int = 512,
) -> np.ndarray:
    """Return the electric-field phasor at the focus ``E(0)`` (V/m, shape ``(3,)``).

    This is the Debye integral evaluated at ``r = 0``, where all rays interfere in
    phase; it is much cheaper than evaluating the full Cartesian grid and is enough for
    the peak-field / ``chi_max`` sweep.
    """
    k = 2.0 * np.pi / lambda0_m
    theta_max = focus.f_number_to_theta_max(f_number)

    th_edges = np.linspace(0.0, theta_max, ntheta + 1)
    th = 0.5 * (th_edges[:-1] + th_edges[1:])
    ph = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(TH) * np.cos(PH), -np.cos(TH), np.sin(TH) * np.sin(PH)], axis=-1)
    far = far_field(n)
    w = np.sin(TH) * (th_edges[1] - th_edges[0]) * (2.0 * np.pi / nphi)

    I2 = float(np.sum(np.sum(np.abs(far) ** 2, axis=-1) * w))
    if I2 <= 0.0:
        raise ValueError("far-field pattern carries zero power over the aperture")
    Cscale = np.sqrt(2.0 * C.ETA_0 * power_w / I2)

    E0 = (-1j * Cscale / lambda0_m) * np.sum(w[..., None] * far, axis=(0, 1))
    return E0


def kappa_eff_vs_f_number(
    f_numbers,
    *,
    lambda0_m: float = 1.0e-6,
    power_w: float = 1.0e15,
    energy_gev: float = 1.0,
    far_field=None,
    ntheta: int = 256,
    nphi: int = 512,
) -> list[dict]:
    """Compute ``chi_max`` and ``a_max`` for each f-number (the Fig. 4 curve).

    Returns a list of dicts with keys ``f_number``, ``chi_max_per_GeV``, ``a_max``.
    """
    if far_field is None:
        far_field = focus.bi_dipole_far_field

    omega = C.omega_from_lambda(lambda0_m)
    E_rel = C.relativistic_field_scale(omega)
    gamma = C.energy_gev_to_gamma(energy_gev)

    out = []
    for fn in f_numbers:
        E0 = focal_field_vector(
            far_field, f_number=fn, lambda0_m=lambda0_m, power_w=power_w,
            ntheta=ntheta, nphi=nphi,
        )
        a_max = float(np.max(np.abs(E0)) / E_rel)
        e_perp = float(np.sqrt(np.abs(E0[0]) ** 2 + np.abs(E0[2]) ** 2))
        chi_max_per_gev = 2.0 * gamma * e_perp / C.E_CRIT / energy_gev
        out.append({"f_number": float(fn), "chi_max_per_GeV": chi_max_per_gev, "a_max": a_max})
    return out


def default_f_number_sweep() -> np.ndarray:
    """The f-number range used in Fig. 4 (0.25 .. 4, log-spaced)."""
    return np.concatenate(([0.25], np.geomspace(0.35, 4.0, 12)))
