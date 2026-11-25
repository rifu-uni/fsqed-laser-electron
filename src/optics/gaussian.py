"""Gaussian source model and paraxial shaping pipeline (before the parabolic mirror).

Implements README items 1, 4 and 5 for ``src/optics``:

* a collimated Gaussian source ``U(R) = exp(-(R/w0)^2)``;
* a thin lens ``t(R) = exp(-i k R^2 / (2 f))``;
* an apodizer (hard aperture of radius ``R_max``, or a super-Gaussian);
* paraxial angular-spectrum (FFT) propagation
  ``U' = F^-1{ F{U} * exp(-i dz (kx^2 + kz^2) / (2 k)) }``;
* apodizer transmission / power accounting.

This pipeline shapes the beam *before* the mirror only; the focal field is obtained
with :mod:`optics.focus` (the vectorial Debye integral), never with scalar paraxial
propagation past the parabola.
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def gaussian_source(R: np.ndarray, w0: float) -> np.ndarray:
    """Collimated Gaussian field amplitude ``exp(-(R/w0)^2)`` (1/e field radius ``w0``)."""
    return np.exp(-((R / w0) ** 2))


def thin_lens_phase(R: np.ndarray, focal_length_m: float, lambda0_m: float) -> np.ndarray:
    """Thin-lens phase ``exp(-i k R^2 / (2 f))``."""
    k = 2.0 * np.pi / lambda0_m
    return np.exp(-1j * k * R * R / (2.0 * focal_length_m))


def apodizer(R: np.ndarray, R_max: float, order: float = 1.0) -> np.ndarray:
    """Apodizer transmission (hard aperture for ``order`` large, super-Gaussian otherwise)."""
    if order <= 0.0:
        raise ValueError("order must be positive")
    return np.exp(-((R / R_max) ** (2.0 * order)))


def propagate_angular_spectrum(
    U: np.ndarray, dx: float, dz: float, dz_prop: float, lambda0_m: float
) -> np.ndarray:
    """Paraxial FFT propagation ``U' = F^-1{ F{U} exp(-i dz (kx^2+kz^2)/(2k)) }``.

    ``U`` has shape ``(nz, nx)`` (a transverse slice).  The returned array is the field
    propagated by ``dz_prop`` along the optical axis (here ``y``).
    """
    nz, nx = U.shape
    k = 2.0 * np.pi / lambda0_m
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    KX, KZ = np.meshgrid(kx, kz, indexing="xy")
    transfer = np.exp(-1j * dz_prop * (KX * KX + KZ * KZ) / (2.0 * k))
    return np.fft.ifft2(np.fft.fft2(U) * transfer)


def beam_at_mirror(
    *,
    R: np.ndarray,
    lambda0_m: float,
    focal_length_m: float,
    w0: float,
    R_max: float,
    apod_order: float = 8.0,
) -> tuple[np.ndarray, float]:
    """Shape a Gaussian beam with the thin lens + apodizer and return the mirror field.

    Returns
    -------
    U_shaped : np.ndarray
        Complex field amplitude at the mirror (before reflection).
    transmission : float
        Apodizer power transmission ``P_transmitted / P_incident`` (power accounting).
    """
    U = gaussian_source(R, w0)
    U = U * thin_lens_phase(R, focal_length_m, lambda0_m)
    T = apodizer(R, R_max, order=apod_order)

    # Power accounting (paraxial: power ~ integral |U|^2 R dR).
    p_in = float(np.trapezoid(np.abs(U) ** 2 * R, R)) if len(R) > 1 else float(np.sum(np.abs(U) ** 2 * R))
    p_out = float(np.trapezoid(np.abs(U * T) ** 2 * R, R)) if len(R) > 1 else float(np.sum(np.abs(U * T) ** 2 * R))
    transmission = p_out / p_in if p_in > 0.0 else 0.0

    return U * T, transmission


def r_max_from_f_number(focal_length_m: float, f_number: float) -> float:
    """Mirror radius ``R_max = L / (2 f-number)`` (the paper's f-number convention)."""
    return focal_length_m / (2.0 * f_number)
