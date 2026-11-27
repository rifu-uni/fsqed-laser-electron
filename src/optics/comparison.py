"""Quantitative comparison between the realistic and ideal incident-beam profiles.

The analytic Sheppard-Larkin profile ``I^p(R)`` (eq. 13) is the *ideal* reference
baseline; the Gaussian-beam x apodizer profile is the *realistic* simulation input.
This module computes the mean squared error (MSE), the Pearson correlation and the
apodizer power transmission between the two.
"""

from __future__ import annotations

import numpy as np

from . import gaussian


def incident_intensity_profiles(*, f_number, w0_over_rmax=0.7, apod_order=8.0, nr=512):
    """Return ``(R, I_ideal, I_real)`` over the mirror aperture (R in units of R_max)."""
    R = np.linspace(0.0, 1.0, nr)
    a_ideal = gaussian.ideal_incident_amplitude(R, f_number=f_number)
    a_real = gaussian.gaussian_incident_amplitude(
        R, w0_over_rmax=w0_over_rmax, apod_order=apod_order
    )
    return R, a_ideal ** 2, a_real ** 2


def _normalize_power(I, R):
    tot = np.trapezoid(I * R, R)
    return I / tot if tot > 0.0 else I


def compare_profiles(I_ideal, I_real, R):
    """Return ``{"mse": ..., "correlation": ...}`` for two intensity profiles.

    Both profiles are normalized to unit total power over the aperture first.
    """
    Ii = _normalize_power(I_ideal, R)
    Ir = _normalize_power(I_real, R)
    mse = float(np.mean((Ii - Ir) ** 2))
    if np.std(Ii) > 0.0 and np.std(Ir) > 0.0:
        corr = float(np.corrcoef(Ii, Ir)[0, 1])
    else:
        corr = 0.0
    return {"mse": mse, "correlation": corr}


def aperture_transmission(w0_over_rmax: float) -> float:
    """Fraction of collimated Gaussian beam power enclosed within ``r <= 1``."""
    if not np.isfinite(w0_over_rmax) or w0_over_rmax <= 0.0:
        raise ValueError(f"w0_over_rmax must be finite and positive, got {w0_over_rmax}")
    return float(1.0 - np.exp(-2.0 / (w0_over_rmax ** 2)))


def apodizer_transmission(w0_over_rmax, apod_order, nr=512):
    """Apodizer power transmission ``P_after / P_in_aperture`` for the Gaussian beam."""
    if not np.isfinite(w0_over_rmax) or w0_over_rmax <= 0.0:
        raise ValueError(f"w0_over_rmax must be finite and positive, got {w0_over_rmax}")
    if not np.isfinite(apod_order) or apod_order <= 0.0:
        raise ValueError(f"apod_order must be finite and positive, got {apod_order}")
    if int(nr) < 2:
        raise ValueError(f"nr must be >= 2, got {nr}")
    R = np.linspace(0.0, 1.0, nr)
    gauss2 = gaussian.gaussian_source(R, w0_over_rmax) ** 2
    apod2 = gaussian.apodizer(R, 1.0, order=apod_order) ** 2
    p_in = np.trapezoid(gauss2 * R, R)
    p_out = np.trapezoid(gauss2 * apod2 * R, R)
    return float(p_out / p_in) if p_in > 0.0 else 0.0


def total_transmission(w0_over_rmax: float, apod_order: float, nr: int = 512) -> float:
    """Total source-to-focus power transmission (aperture truncation x apodizer)."""
    return float(
        aperture_transmission(w0_over_rmax)
        * apodizer_transmission(w0_over_rmax, apod_order, nr=nr)
    )


def summarize_degradation(*, f_number, w0_over_rmax=0.7, apod_order=8.0, nr=512):
    """Full quantitative degradation report of the realistic vs ideal profile."""
    R, I_ideal, I_real = incident_intensity_profiles(
        f_number=f_number, w0_over_rmax=w0_over_rmax, apod_order=apod_order, nr=nr
    )
    result = compare_profiles(I_ideal, I_real, R)
    ap_t = aperture_transmission(w0_over_rmax)
    apod_t = apodizer_transmission(w0_over_rmax, apod_order, nr)
    result["transmission"] = apod_t
    result["aperture_transmission"] = ap_t
    result["total_transmission"] = float(ap_t * apod_t)
    result["f_number"] = float(f_number)
    return result