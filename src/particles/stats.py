"""Shot-count statistics, electron energy-loss comparison, and Fig. 4 / Fig. 5 benchmark curves."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import (
    C_LIGHT,
    E_CHARGE,
    E_CRIT,
    KAPPA_BIDIPOLE_MAX,
    energy_gev_to_gamma,
)


@dataclass(frozen=True)
class ShotEstimate:
    """3-sigma shot-count statistics for photon-based SFQED detection."""

    sigma_cm2: float
    eta: float
    n_e_cm3: float
    tau_l_s: float
    n_signal_per_shot: float
    n_total_per_shot: float
    n_total_required: float
    n_shots_required: float


def estimate_shots_for_confidence(
    sigma_cm2: float,
    eta: float,
    n_e_cm3: float = 1.0e19,
    tau_l_s: float = 13.34256e-15,
    rel_accuracy: float = 0.01,
    confidence_sigma: float = 3.0,
) -> ShotEstimate:
    """Estimate shots required to detect a relative rate deviation at given confidence level."""
    for name, val in (
        ("sigma_cm2", sigma_cm2),
        ("eta", eta),
        ("n_e_cm3", n_e_cm3),
        ("tau_l_s", tau_l_s),
        ("rel_accuracy", rel_accuracy),
        ("confidence_sigma", confidence_sigma),
    ):
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {val}")
    if eta > 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
    if rel_accuracy >= 1.0:
        raise ValueError(f"rel_accuracy must be in (0, 1), got {rel_accuracy}")

    c_cm_s = 100.0 * C_LIGHT
    n_total_per_shot = sigma_cm2 * n_e_cm3 * tau_l_s * c_cm_s
    n_signal_per_shot = eta * n_total_per_shot
    n_total_required = (confidence_sigma / (rel_accuracy * eta)) ** 2
    n_shots_required = n_total_required / n_total_per_shot

    return ShotEstimate(
        sigma_cm2=float(sigma_cm2),
        eta=float(eta),
        n_e_cm3=float(n_e_cm3),
        tau_l_s=float(tau_l_s),
        n_signal_per_shot=float(n_signal_per_shot),
        n_total_per_shot=float(n_total_per_shot),
        n_total_required=float(n_total_required),
        n_shots_required=float(n_shots_required),
    )


def estimate_electron_loss_shots(
    chi: float = 10.0,
    a_max: float = 200.0,
    prop_length_lambda: float = 2.0,
    bunch_charge_pc: float = 100.0,
    rel_accuracy: float = 0.01,
    confidence_sigma: float = 3.0,
) -> dict[str, float]:
    """Estimate shots needed when using electron energy loss instead of emitted photons (Sec. V)."""
    for name, val in (
        ("chi", chi),
        ("a_max", a_max),
        ("prop_length_lambda", prop_length_lambda),
        ("bunch_charge_pc", bunch_charge_pc),
        ("rel_accuracy", rel_accuracy),
        ("confidence_sigma", confidence_sigma),
    ):
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {val}")
    if rel_accuracy >= 1.0:
        raise ValueError(f"rel_accuracy must be in (0, 1), got {rel_accuracy}")

    mfp_over_lambda = 15.0 * (chi ** (1.0 / 3.0)) / a_max
    survival_fraction = math.exp(-prop_length_lambda / mfp_over_lambda)
    electrons_required = (confidence_sigma / (rel_accuracy * survival_fraction)) ** 2
    electrons_per_bunch = (bunch_charge_pc * 1.0e-12) / E_CHARGE
    shots_required = electrons_required / electrons_per_bunch

    mfp_paper = 0.15 * ((chi / 10.0) ** (1.0 / 3.0)) * (200.0 / a_max)
    survival_paper = math.exp(-prop_length_lambda / mfp_paper)
    electrons_required_paper = (confidence_sigma / (rel_accuracy * survival_paper)) ** 2
    shots_required_paper = electrons_required_paper / electrons_per_bunch

    return {
        "mfp_over_lambda": float(mfp_over_lambda),
        "survival_fraction": float(survival_fraction),
        "electrons_required": float(electrons_required),
        "electrons_per_bunch": float(electrons_per_bunch),
        "shots_required": float(shots_required),
        "shots_required_paper_rounded": float(shots_required_paper),
    }


def f_number_to_opening_angle_deg(
    f_number: float | np.ndarray,
) -> np.ndarray | float:
    """Convert parabolic mirror f-number = L / (2 R_max) to cone opening angle zeta (deg)."""
    f_arr = np.asarray(f_number, dtype=np.float64)
    if not np.all(np.isfinite(f_arr)) or np.any(f_arr <= 0.0):
        raise ValueError("f_number must contain only finite positive values")

    rho = 1.0 / (2.0 * f_arr)
    zeta_rad = 0.5 * math.pi + np.arctan(0.25 * rho - 1.0 / rho)
    zeta_deg = np.degrees(zeta_rad)
    if np.ndim(f_number) == 0:
        return float(zeta_deg)
    return zeta_deg


def fig4_reference_curve(
    f_numbers: np.ndarray | None = None,
    power_w: float = 1.0e15,
    energy_gev: float = 1.0,
    lambda_m: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Compute theoretical truncated-bidipole curve and digitized Fig. 4 benchmark values."""
    if not math.isfinite(power_w) or power_w <= 0.0:
        raise ValueError(f"power_w must be finite and positive, got {power_w}")
    if not math.isfinite(energy_gev) or energy_gev <= 0.0:
        raise ValueError(f"energy_gev must be finite and positive, got {energy_gev}")
    if not math.isfinite(lambda_m) or lambda_m <= 0.0:
        raise ValueError(f"lambda_m must be finite and positive, got {lambda_m}")

    f_grid = (
        np.linspace(0.01, 2.0, 200, dtype=np.float64)
        if f_numbers is None
        else np.asarray(f_numbers, dtype=np.float64)
    )
    if f_grid.ndim != 1 or f_grid.size == 0 or not np.all(np.isfinite(f_grid)) or np.any(f_grid <= 0.0):
        raise ValueError("f_numbers must be a non-empty 1D array of finite positive numbers")

    opening_angle_deg = np.asarray(f_number_to_opening_angle_deg(f_grid), dtype=np.float64)
    zeta_rad = np.radians(opening_angle_deg)
    power_frac = np.maximum(0.0, (8.0 - (1.0 + np.cos(zeta_rad)) ** 3) / 8.0)
    kappa = KAPPA_BIDIPOLE_MAX * np.sqrt(power_frac)

    scale = energy_gev * math.sqrt(power_w / 1.0e15) * (1.0e-6 / lambda_m)
    chi_max = kappa * scale
    gamma = energy_gev_to_gamma(energy_gev) if energy_gev > 0.001 else energy_gev / 0.00051099895
    field_strength_vm = (chi_max * E_CRIT) / (2.0 * gamma)
    field_strength_1e14_vm = field_strength_vm / 1.0e14

    digitized_f = np.array([0.01, 0.25, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    digitized_zeta_rad = np.radians(np.asarray(f_number_to_opening_angle_deg(digitized_f)))
    digitized_kappa = KAPPA_BIDIPOLE_MAX * np.sqrt(
        np.maximum(0.0, (8.0 - (1.0 + np.cos(digitized_zeta_rad)) ** 3) / 8.0)
    )
    digitized_chi_max = digitized_kappa * scale

    return {
        "f_numbers": f_grid,
        "opening_angle_deg": opening_angle_deg,
        "kappa": kappa,
        "chi_max": chi_max,
        "field_strength_vm": field_strength_vm,
        "field_strength_1e14_vm": field_strength_1e14_vm,
        "digitized_f_numbers": digitized_f,
        "digitized_chi_max": digitized_chi_max,
    }


def fig5_reference_contours(
    energy_gev_grid: np.ndarray | None = None,
    power_w_grid: np.ndarray | None = None,
    lambda_m: float = 0.8e-6,
    kappa: float = 5.10,
) -> dict[str, np.ndarray]:
    """Compute chi_max / 2 = {1, 10, 100, 1000} contour lines across (energy, power) space."""
    if not math.isfinite(lambda_m) or lambda_m <= 0.0:
        raise ValueError(f"lambda_m must be finite and positive, got {lambda_m}")
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError(f"kappa must be finite and positive, got {kappa}")

    energies = (
        np.geomspace(1.0, 200.0, 100, dtype=np.float64)
        if energy_gev_grid is None
        else np.asarray(energy_gev_grid, dtype=np.float64)
    )
    powers = (
        np.geomspace(1.0e12, 1.0e17, 100, dtype=np.float64)
        if power_w_grid is None
        else np.asarray(power_w_grid, dtype=np.float64)
    )
    if energies.ndim != 1 or energies.size == 0 or not np.all(np.isfinite(energies)) or np.any(energies <= 0.0):
        raise ValueError("energy_gev_grid must be a non-empty 1D array of positive values")
    if powers.ndim != 1 or powers.size == 0 or not np.all(np.isfinite(powers)) or np.any(powers <= 0.0):
        raise ValueError("power_w_grid must be a non-empty 1D array of positive values")

    ee, pp = np.meshgrid(energies, powers, indexing="xy")
    chi_max_grid = kappa * ee * np.sqrt(pp / 1.0e15) * (1.0e-6 / lambda_m)
    chi_half_grid = 0.5 * chi_max_grid

    levels = np.array([1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
    contour_power_w = np.empty((levels.size, energies.size), dtype=np.float64)
    for idx, level in enumerate(levels):
        # 0.5 * kappa * E * sqrt(P / 1e15) * (1e-6 / lambda) = level
        contour_power_w[idx] = 1.0e15 * ((2.0 * level * lambda_m) / (1.0e-6 * kappa * energies)) ** 2

    return {
        "energy_gev": energies,
        "power_w": powers,
        "chi_max_grid": chi_max_grid,
        "chi_half_grid": chi_half_grid,
        "levels": levels,
        "contour_power_w": contour_power_w,
    }
