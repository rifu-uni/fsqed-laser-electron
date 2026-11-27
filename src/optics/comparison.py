"""Quantitative comparison between the realistic and ideal incident-beam profiles.

The analytic Sheppard-Larkin profile ``I^p(R)`` (eq. 13) is the *ideal* reference
baseline; the Gaussian-beam x apodizer profile is the *realistic* simulation input.
This module computes the circular-disk area-weighted mean squared error (MSE), root
mean squared error (RMSE), Pearson correlation, and power transmission between the two.
"""

from __future__ import annotations

import numpy as np

from . import gaussian


def _validate_radial_grid(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Validate a strictly increasing non-negative 1-D radial grid and return its area norm."""
    r_arr = np.asarray(R, dtype=np.float64)
    if (
        r_arr.ndim != 1
        or r_arr.size < 2
        or not np.all(np.isfinite(r_arr))
        or np.any(r_arr < 0.0)
        or np.any(np.diff(r_arr) <= 0.0)
    ):
        raise ValueError("R must be a finite, strictly increasing, non-negative 1D array with >= 2 points")
    area_norm = float(np.trapezoid(r_arr, r_arr))
    if area_norm <= 0.0:
        raise ValueError("Radial grid must span a positive aperture area")
    return r_arr, area_norm


def _area_mean(values: np.ndarray, r_arr: np.ndarray, area_norm: float) -> float:
    """Compute the circular-aperture area-weighted mean ``int values * R dR / int R dR``."""
    return float(np.trapezoid(values * r_arr, r_arr) / area_norm)


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
    if tot <= 0.0:
        raise ValueError("Intensity profile has non-positive total radial power")
    return I / tot


def compare_profiles(I_ideal, I_real, R):
    """Return area-weighted ``{"mse", "rmse", "relative_rmse", "correlation"}`` for two
    intensity profiles.

    Both profiles are normalized to unit total radial power (``int I(R) R dR = 1``)
    first, and all moments are weighted by ``R dR`` over the circular mirror aperture.
    ``relative_rmse`` is ``rmse / mean(I_ideal)`` after that normalization (``0.0``
    when the normalized ideal profile has zero mean).
    """
    r_arr, area_norm = _validate_radial_grid(R)
    i_ideal_arr = np.asarray(I_ideal, dtype=np.float64)
    i_real_arr = np.asarray(I_real, dtype=np.float64)
    if i_ideal_arr.shape != r_arr.shape or i_real_arr.shape != r_arr.shape:
        raise ValueError("I_ideal and I_real must have the same 1D shape as R")
    if not np.all(np.isfinite(i_ideal_arr)) or not np.all(np.isfinite(i_real_arr)):
        raise ValueError("Intensity profiles must contain only finite values")

    Ii = _normalize_power(i_ideal_arr, r_arr)
    Ir = _normalize_power(i_real_arr, r_arr)
    diff = Ii - Ir
    mse = _area_mean(diff * diff, r_arr, area_norm)
    rmse = float(np.sqrt(max(mse, 0.0)))

    mi = _area_mean(Ii, r_arr, area_norm)
    mr = _area_mean(Ir, r_arr, area_norm)
    di = Ii - mi
    dr = Ir - mr
    vi = _area_mean(di * di, r_arr, area_norm)
    vr = _area_mean(dr * dr, r_arr, area_norm)
    if vi > 0.0 and vr > 0.0:
        corr = float(np.clip(_area_mean(di * dr, r_arr, area_norm) / np.sqrt(vi * vr), -1.0, 1.0))
    else:
        corr = 0.0
    relative_rmse = float(rmse / mi) if mi > 0.0 else 0.0
    return {"mse": mse, "rmse": rmse, "relative_rmse": relative_rmse, "correlation": corr}


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
    """Full quantitative degradation report of the realistic vs ideal profile.

    ``transmission`` is the apodizer-only transmission (in-aperture), while
    ``total_transmission`` also includes the finite-aperture truncation of the
    collimated Gaussian beam.
    """
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