"""Quantum nonlinearity parameter chi, electron deviation angle, and Fig. 2 phase-space diagnostics."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .constants import C_LIGHT, E_CRIT, MC_KGMS


@njit(cache=True)
def compute_chi_numba(
    E_vm: np.ndarray,
    B_t: np.ndarray,
    p_kgms: np.ndarray,
) -> float:
    """Compute dimensionless quantum nonlinearity parameter chi for a single electron."""
    ux = p_kgms[0] / MC_KGMS
    uy = p_kgms[1] / MC_KGMS
    uz = p_kgms[2] / MC_KGMS

    gamma = math.sqrt(1.0 + ux * ux + uy * uy + uz * uz)
    bx = ux / gamma
    by = uy / gamma
    bz = uz / gamma

    # F_perp = E + c * (beta x B)
    fx = E_vm[0] + C_LIGHT * (by * B_t[2] - bz * B_t[1])
    fy = E_vm[1] + C_LIGHT * (bz * B_t[0] - bx * B_t[2])
    fz = E_vm[2] + C_LIGHT * (bx * B_t[1] - by * B_t[0])

    f_par = E_vm[0] * bx + E_vm[1] * by + E_vm[2] * bz
    rad_sq = fx * fx + fy * fy + fz * fz - f_par * f_par
    if rad_sq <= 0.0:
        return 0.0
    return (gamma / E_CRIT) * math.sqrt(rad_sq)


def compute_chi(
    E_vm: np.ndarray,
    B_t: np.ndarray,
    p_kgms: np.ndarray,
) -> np.ndarray | float:
    """Vectorized quantum nonlinearity parameter chi for (..., 3) field and momentum arrays."""
    e_arr = np.asarray(E_vm, dtype=np.float64)
    b_arr = np.asarray(B_t, dtype=np.float64)
    p_arr = np.asarray(p_kgms, dtype=np.float64)

    if e_arr.ndim == 0 or b_arr.ndim == 0 or p_arr.ndim == 0:
        raise ValueError("E_vm, B_t, and p_kgms must have at least 1 dimension of length 3")
    if e_arr.shape[-1] != 3 or b_arr.shape[-1] != 3 or p_arr.shape[-1] != 3:
        raise ValueError("Last axis of E_vm, B_t, and p_kgms must have length 3")
    if not (
        np.all(np.isfinite(e_arr)) and np.all(np.isfinite(b_arr)) and np.all(np.isfinite(p_arr))
    ):
        raise ValueError("E_vm, B_t, and p_kgms must contain only finite values")

    e_b, b_b, p_b = np.broadcast_arrays(e_arr, b_arr, p_arr)
    u = p_b / MC_KGMS
    gamma = np.sqrt(1.0 + np.sum(u * u, axis=-1))
    beta = u / gamma[..., None]

    f_perp = e_b + C_LIGHT * np.cross(beta, b_b, axis=-1)
    f_par = np.sum(e_b * beta, axis=-1)
    rad_sq = np.maximum(0.0, np.sum(f_perp * f_perp, axis=-1) - f_par * f_par)
    chi = (gamma / E_CRIT) * np.sqrt(rad_sq)

    if e_arr.ndim == 1 and b_arr.ndim == 1 and p_arr.ndim == 1:
        return float(chi)
    return chi


@njit(cache=True)
def compute_deviation_angle_numba(
    p_kgms: np.ndarray,
    signed_z: bool = False,
) -> float:
    """Compute electron deviation angle (rad) relative to +y propagation axis."""
    px = p_kgms[0]
    py = p_kgms[1]
    pz = p_kgms[2]
    if px == 0.0 and py == 0.0 and pz == 0.0:
        return 0.0
    if signed_z:
        return math.atan2(pz, py)
    return math.atan2(math.hypot(px, pz), py)


def compute_deviation_angle(
    p_kgms: np.ndarray,
    signed_z: bool = False,
) -> np.ndarray | float:
    """Vectorized electron deviation angle (rad) relative to +y for (..., 3) momentum."""
    p_arr = np.asarray(p_kgms, dtype=np.float64)
    if p_arr.ndim == 0 or p_arr.shape[-1] != 3:
        raise ValueError(f"p_kgms must have shape (..., 3), got {p_arr.shape}")
    if not np.all(np.isfinite(p_arr)):
        raise ValueError("p_kgms must contain only finite values")

    px = p_arr[..., 0]
    py = p_arr[..., 1]
    pz = p_arr[..., 2]
    is_zero = (px == 0.0) & (py == 0.0) & (pz == 0.0)

    if signed_z:
        angle = np.arctan2(pz, py)
    else:
        angle = np.arctan2(np.hypot(px, pz), py)
    angle = np.where(is_zero, 0.0, angle)

    if p_arr.ndim == 1:
        return float(angle)
    return angle


def compute_high_chi_time_fraction(
    chi_trace: np.ndarray,
    dt_s: float,
    chi_max: float | None = None,
) -> np.ndarray | float:
    """Compute high-chi residence time fraction F = (1/T) int Theta(chi - chi_max/2) dt."""
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be finite and positive, got {dt_s}")

    chi_arr = np.asarray(chi_trace, dtype=np.float64)
    if chi_arr.ndim not in (1, 2) or chi_arr.shape[-1] == 0:
        raise ValueError(f"chi_trace must be non-empty 1D or 2D, got shape {chi_arr.shape}")
    if not np.all(np.isfinite(chi_arr)) or np.any(chi_arr < 0.0):
        raise ValueError("chi_trace must contain finite non-negative values")

    if chi_max is None:
        chi_max_eff = float(np.max(chi_arr))
    else:
        if not math.isfinite(chi_max) or chi_max <= 0.0:
            raise ValueError(f"chi_max must be finite and positive, got {chi_max}")
        chi_max_eff = float(chi_max)

    if chi_max_eff == 0.0:
        if chi_arr.ndim == 1:
            return 0.0
        return np.zeros(chi_arr.shape[0], dtype=np.float64)

    frac = np.mean(chi_arr > 0.5 * chi_max_eff, axis=-1, dtype=np.float64)
    if chi_arr.ndim == 1:
        return float(frac)
    return frac


def compute_fig2_phase_space(
    alpha_traces: np.ndarray,
    chi_traces: np.ndarray,
    a_max: float,
    gamma0: float,
    chi_max: float,
    signed_alpha: bool = False,
    n_bins_alpha: int = 120,
    n_bins_chi: int = 120,
) -> dict[str, np.ndarray]:
    """Compute normalized 2D residence-time density d^2f/(du_alpha du_chi) matching Fig. 2."""
    alpha_arr = np.asarray(alpha_traces, dtype=np.float64)
    chi_arr = np.asarray(chi_traces, dtype=np.float64)

    if alpha_arr.shape != chi_arr.shape or alpha_arr.size == 0:
        raise ValueError(
            f"alpha_traces {alpha_arr.shape} and chi_traces {chi_arr.shape} must be non-empty and match"
        )
    if not (np.all(np.isfinite(alpha_arr)) and np.all(np.isfinite(chi_arr))):
        raise ValueError("alpha_traces and chi_traces must contain finite values")
    if not math.isfinite(a_max) or a_max <= 0.0:
        raise ValueError(f"a_max must be finite and positive, got {a_max}")
    if not math.isfinite(gamma0) or gamma0 <= 1.0:
        raise ValueError(f"gamma0 must be finite and > 1, got {gamma0}")
    if not math.isfinite(chi_max) or chi_max <= 0.0:
        raise ValueError(f"chi_max must be finite and positive, got {chi_max}")
    if n_bins_alpha < 2 or n_bins_chi < 2:
        raise ValueError("n_bins_alpha and n_bins_chi must be >= 2")

    alpha_scale = a_max / gamma0
    u_alpha = alpha_arr.ravel() / alpha_scale
    u_chi = chi_arr.ravel() / chi_max

    alpha_bound = max(1.1, float(np.max(np.abs(u_alpha))) * 1.02)
    alpha_range = (-alpha_bound, alpha_bound) if signed_alpha else (0.0, alpha_bound)
    chi_upper = max(1.05, float(np.max(u_chi)) * 1.02)
    chi_range = (0.0, chi_upper)

    density, u_alpha_edges, u_chi_edges = np.histogram2d(
        u_alpha,
        u_chi,
        bins=(n_bins_alpha, n_bins_chi),
        range=(alpha_range, chi_range),
        density=True,
    )
    # Transpose to shape (n_bins_chi, n_bins_alpha) so axis 0 is u_chi and axis 1 is u_alpha
    density_yx = np.ascontiguousarray(density.T, dtype=np.float64)
    log_density = np.full_like(density_yx, np.nan)
    pos = density_yx > 0.0
    log_density[pos] = np.log(density_yx[pos])

    return {
        "density": density_yx,
        "log_density": log_density,
        "u_alpha_edges": u_alpha_edges,
        "u_chi_edges": u_chi_edges,
        "u_alpha_centers": 0.5 * (u_alpha_edges[:-1] + u_alpha_edges[1:]),
        "u_chi_centers": 0.5 * (u_chi_edges[:-1] + u_chi_edges[1:]),
    }
