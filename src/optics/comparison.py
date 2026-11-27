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


def apodizer_transmission(w0_over_rmax, apod_order, nr=512):
    """Apodizer power transmission ``P_after / P_before`` for the Gaussian beam."""
    R = np.linspace(0.0, 1.0, nr)
    gauss2 = gaussian.gaussian_source(R, w0_over_rmax) ** 2
    apod2 = gaussian.apodizer(R, 1.0, order=apod_order) ** 2
    p_in = np.trapezoid(gauss2 * R, R)
    p_out = np.trapezoid(gauss2 * apod2 * R, R)
    return float(p_out / p_in) if p_in > 0.0 else 0.0


def summarize_degradation(*, f_number, w0_over_rmax=0.7, apod_order=8.0, nr=512):
    """Full quantitative degradation report of the realistic vs ideal profile."""
    R, I_ideal, I_real = incident_intensity_profiles(
        f_number=f_number, w0_over_rmax=w0_over_rmax, apod_order=apod_order, nr=nr
    )
    result = compare_profiles(I_ideal, I_real, R)
    result["transmission"] = apodizer_transmission(w0_over_rmax, apod_order, nr)
    result["f_number"] = float(f_number)
    return result