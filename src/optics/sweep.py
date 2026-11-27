"""``kappa_eff(f-number)`` sweep -> comparison with Fig. 4 of Olofsson & Gonoskov.

Fig. 4 plots ``chi_max`` as a function of the parabolic-mirror f-number for
``P = 1 PW``, ``eps = 1 GeV`` and ``lambda = 1 um``, with the peak electric/magnetic
field on the right axis.

The far-field model is supplied through a *builder* ``far_field_builder(f_number) ->
callable(n)`` because the realistic (Gaussian + apodizer + mirror) far field depends on
the f-number (which defines the mirror aperture ``theta_max``).  Two builders are
provided: :func:`ideal_far_field_builder` (analytic Sheppard-Larkin baseline) and
:func:`gaussian_far_field_builder` (realistic optical pipeline).

``chi_max`` scales as ``chi ~ eps * lambda^-1 * P^{1/2}``; ``a0`` (the field strength in
relativistic units) scales as ``P^{1/2}`` and is wavelength-independent for a fixed
f-number.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import comparison
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


def ideal_far_field_builder(pol: str = "z-linear"):
    """Builder for the analytic Sheppard-Larkin baseline (independent of f-number)."""
    focus._validate_pol(pol)

    def build(f_number):
        return lambda n: focus.bi_dipole_far_field(n, pol=pol)

    return build


def gaussian_far_field_builder(
    w0_over_rmax: float = 0.7, apod_order: float = 8.0, pol: str = "z-linear"
):
    """Builder for the realistic Gaussian + apodizer + mirror pipeline."""
    focus._validate_pol(pol)

    def build(f_number):
        def far(n):
            return focus.gaussian_far_field(
                n,
                f_number=f_number,
                w0_over_rmax=w0_over_rmax,
                apod_order=apod_order,
                pol=pol,
            )

        return far

    return build


def gaussian_transmission(
    w0_over_rmax: float = 0.7,
    apod_order: float = 8.0,
    *,
    include_aperture_truncation: bool = True,
    nr: int = 512,
) -> float:
    """Return source-to-focus power transmission for the Gaussian + apodizer system."""
    if include_aperture_truncation:
        return comparison.total_transmission(w0_over_rmax, apod_order, nr=nr)
    return comparison.apodizer_transmission(w0_over_rmax, apod_order, nr=nr)


def kappa_eff_vs_f_number(
    f_numbers,
    *,
    lambda0_m: float = 1.0e-6,
    power_w: float = 1.0e15,
    energy_gev: float = 1.0,
    far_field_builder=None,
    power_transmission: float | Callable[[float], float] = 1.0,
    ntheta: int = 256,
    nphi: int = 512,
) -> list[dict]:
    """Compute ``chi_max``, ``a_max``, and ``kappa_eff`` for each f-number (Fig. 4).

    Parameters
    ----------
    far_field_builder : callable, optional
        ``far_field_builder(f_number) -> far_field(n)``.  Defaults to the ideal
        Sheppard-Larkin baseline.
    power_transmission : float or callable, optional
        Fraction ``T in (0, 1]`` of source ``power_w`` delivered through the
        aperture/apodizer to the mirror focus (or ``callable(f_number) -> T``).
        Defaults to ``1.0`` (lossless).

    Returns a list of dicts with keys ``f_number``, ``theta_max_deg``,
    ``chi_max_per_GeV``, ``a_max``, ``kappa_eff``, ``transmission``.
    """
    if far_field_builder is None:
        far_field_builder = ideal_far_field_builder()

    omega = C.omega_from_lambda(lambda0_m)
    E_rel = C.relativistic_field_scale(omega)
    gamma = C.energy_gev_to_gamma(energy_gev)
    eq1_scale = float(np.sqrt(power_w / 1.0e15) * (1.0e-6 / lambda0_m))

    out = []
    for fn in f_numbers:
        t_val = float(power_transmission(fn)) if callable(power_transmission) else float(power_transmission)
        if not np.isfinite(t_val) or t_val <= 0.0 or t_val > 1.0:
            raise ValueError(f"power_transmission must be finite in (0, 1], got {t_val}")

        far_field = far_field_builder(fn)
        E0 = focal_field_vector(
            far_field, f_number=fn, lambda0_m=lambda0_m, power_w=power_w * t_val,
            ntheta=ntheta, nphi=nphi,
        )
        # Exact peak instantaneous real field amplitude over an optical cycle
        # (matches LP amplitude and E_lp / sqrt(2) for circular polarization).
        e_abs2 = float(np.sum(np.abs(E0) ** 2))
        e_dot_abs = float(abs(np.dot(E0, E0)))
        e_peak = float(np.sqrt(0.5 * (e_abs2 + e_dot_abs)))
        a_max = e_peak / E_rel

        e_perp_abs2 = float(abs(E0[0]) ** 2 + abs(E0[2]) ** 2)
        e_perp_dot_abs = float(abs(E0[0] ** 2 + E0[2] ** 2))
        e_perp_peak = float(np.sqrt(0.5 * (e_perp_abs2 + e_perp_dot_abs)))
        chi_max_per_gev = 2.0 * gamma * e_perp_peak / C.E_CRIT / energy_gev
        kappa_eff = chi_max_per_gev / eq1_scale
        theta_max_deg = float(np.degrees(focus.f_number_to_theta_max(float(fn))))
        out.append({
            "f_number": float(fn),
            "theta_max_deg": theta_max_deg,
            "chi_max_per_GeV": chi_max_per_gev,
            "a_max": a_max,
            "kappa_eff": kappa_eff,
            "transmission": t_val,
        })
    return out


def f_number_to_opening_angle_deg(f_number: float | np.ndarray) -> float | np.ndarray:
    """Convert parabolic-mirror f-number ``L/(2 R_max)`` to opening angle (deg)."""
    f_arr = np.asarray(f_number, dtype=np.float64)
    if not np.all(np.isfinite(f_arr)) or np.any(f_arr <= 0.0):
        raise ValueError("f_number must contain only finite positive values")
    deg = np.degrees(2.0 * np.arctan(1.0 / (4.0 * f_arr)))
    return float(deg) if np.ndim(f_number) == 0 else deg


def default_f_number_sweep() -> np.ndarray:
    """The f-number range matching Fig. 4 of Olofsson & Gonoskov (0.02 .. 2.0)."""
    small_f = np.geomspace(0.02, 0.25, 10, endpoint=False)
    mid_f = np.linspace(0.25, 2.0, 15)
    return np.unique(np.concatenate((small_f, mid_f, [0.5, 1.0, 1.5, 2.0])))