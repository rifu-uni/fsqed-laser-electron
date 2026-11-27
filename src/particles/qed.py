"""Locally Constant Field Approximation (LCFA) QED photon emission, recoil, and selection cuts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.integrate import cumulative_trapezoid, quad
from scipy.special import gamma as gamma_func
from scipy.special import kv

from .chi import compute_chi_numba, compute_deviation_angle_numba
from .constants import ALPHA_FS, C_LIGHT, HBAR, MC2_GEV, MC2_J, MC_KGMS
from .fields import FieldPack, evaluate_eb_numba
from .pusher import ElectronBatch, build_simulation_time_grid, step_boris_numba, step_vay_numba

# W0 = alpha_fs * m_e * c^2 / (sqrt(3) * pi * hbar)
W0_LCFA_PER_S: float = (ALPHA_FS * MC2_J) / (math.sqrt(3.0) * math.pi * HBAR)
_ASymptote_G_COEFF: float = 5.0 * math.pi / 2.0  # ~ 7.853981633974483
_INT_K13_TOTAL: float = math.pi / math.sqrt(3.0)  # int_0^inf K_{1/3}(s) ds
_Y_TILDE_CUTOFF: float = 100.0
_INV_CDF_T_MAX: float = 30.0  # 1 - q_max = exp(-30) ~ 9.36e-14


@dataclass(frozen=True)
class LCFATable:
    """Precomputed LCFA synchrotron rate, high-energy fractions, and inverse CDF table."""

    chi_grid: np.ndarray  # (n_chi,) float64
    log_chi_grid: np.ndarray  # (n_chi,) float64
    q_grid: np.ndarray  # (n_q,) float64
    g_rate: np.ndarray  # (n_chi,) float64
    high_energy_fraction: np.ndarray  # (n_chi,) float64, fraction with delta > 0.5
    loss_gt_1pct_fraction: np.ndarray  # (n_chi,) float64, fraction with delta > 0.01
    mean_delta_total: np.ndarray  # (n_chi,) float64, E[delta]
    mean_delta_high: np.ndarray  # (n_chi,) float64, E[delta | delta > 0.5]
    mean_delta_small: np.ndarray  # (n_chi,) float64, E[delta * 1_{delta <= 0.01}]
    mean_delta2_small: np.ndarray  # (n_chi,) float64, E[delta^2 * 1_{delta <= 0.01}]
    inv_cdf: np.ndarray  # (n_chi, n_q) float64, delta(chi, q)


@dataclass(frozen=True)
class PhotonCatalog:
    """Catalog of emitted photons (stochastic Monte Carlo or weighted expectation records)."""

    energy_gev: np.ndarray  # (N_photons,) float64
    delta_init: np.ndarray  # (N_photons,) float64, hbar*omega / (gamma0 * m c^2)
    alpha_rad: np.ndarray  # (N_photons,) float64
    chi_emit: np.ndarray  # (N_photons,) float64
    gamma_prior: np.ndarray  # (N_photons,) float64
    gamma0: np.ndarray  # (N_photons,) float64
    b_impact_m: np.ndarray  # (N_photons,) float64
    weight: np.ndarray  # (N_photons,) float64
    t_emit_s: np.ndarray  # (N_photons,) float64
    particle_idx: np.ndarray  # (N_photons,) int64


@dataclass(frozen=True)
class QEDResult:
    """Combined trajectory traces and emitted photon catalog from simulate_qed_ensemble."""

    t_s: np.ndarray  # (N_steps,) float64
    r_m: np.ndarray  # (N_particles, N_steps, 3) float64
    p_kgms: np.ndarray  # (N_particles, N_steps, 3) float64
    gamma: np.ndarray  # (N_particles, N_steps) float64
    chi: np.ndarray  # (N_particles, N_steps) float64
    alpha: np.ndarray  # (N_particles, N_steps) float64
    p_no_loss: np.ndarray  # (N_particles, N_steps) float64
    photons: PhotonCatalog
    batch: ElectronBatch
    mode: str


def _integral_k13_tail(y_vals: np.ndarray) -> np.ndarray:
    """Evaluate int_{y}^{inf} K_{1/3}(s) ds smoothly over arbitrary positive array y_vals."""
    out = np.zeros_like(y_vals, dtype=np.float64)
    active = y_vals <= _Y_TILDE_CUTOFF
    if not np.any(active):
        return out

    # Tabulate int_y^inf K_{1/3}(s) ds accurately via variable change s = t^(3/2) on [0, y_max]
    # and direct integration from y to inf for moderate/large y.
    y_act = y_vals[active]
    res = np.empty_like(y_act, dtype=np.float64)
    for idx, y in enumerate(y_act):
        if y <= 0.0:
            res[idx] = _INT_K13_TOTAL
        elif y < 1.0:
            t_max = y ** (2.0 / 3.0)
            part, _ = quad(lambda t: 1.5 * math.sqrt(t) * float(kv(1.0 / 3.0, t**1.5)), 0.0, t_max, epsabs=1e-12, epsrel=1e-10)
            res[idx] = max(0.0, _INT_K13_TOTAL - part)
        else:
            part, _ = quad(lambda s: float(kv(1.0 / 3.0, s)), float(y), 120.0, epsabs=1e-12, epsrel=1e-10)
            res[idx] = max(0.0, part)
    out[active] = res
    return out


def _build_k13_tail_interpolator() -> tuple[np.ndarray, np.ndarray]:
    """Precompute a fine lookup grid for int_y^inf K_{1/3}(s) ds on y in [0, 105]."""
    y_grid = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            np.geomspace(1.0e-12, 105.0, 600, dtype=np.float64),
        ]
    )
    vals = _integral_k13_tail(y_grid)
    return y_grid, vals


_K13_Y_GRID, _K13_TAIL_VALS = _build_k13_tail_interpolator()


def _eval_k13_tail(y_vals: np.ndarray) -> np.ndarray:
    return np.interp(y_vals, _K13_Y_GRID, _K13_TAIL_VALS, left=_INT_K13_TOTAL, right=0.0)


def build_lcfa_table(
    chi_min: float = 1.0e-4,
    chi_max: float = 2.0e3,
    n_chi: int = 128,
    n_q: int = 256,
) -> LCFATable:
    """Build precomputed LCFA rate, fraction, and inverse-CDF tables across chi."""
    if not (math.isfinite(chi_min) and math.isfinite(chi_max)) or chi_min <= 0.0 or chi_max <= chi_min:
        raise ValueError(f"Must satisfy 0 < chi_min < chi_max, got ({chi_min}, {chi_max})")
    if n_chi < 8 or n_q < 8:
        raise ValueError("n_chi and n_q must be >= 8")

    chi_grid = np.geomspace(chi_min, chi_max, n_chi, dtype=np.float64)
    log_chi_grid = np.log(chi_grid)
    s_grid = np.linspace(0.0, 1.0, n_q, dtype=np.float64)
    t_grid = _INV_CDF_T_MAX * (s_grid**2)
    q_grid = -np.expm1(-t_grid)

    g_rate = np.empty(n_chi, dtype=np.float64)
    high_energy_fraction = np.empty(n_chi, dtype=np.float64)
    loss_gt_1pct_fraction = np.empty(n_chi, dtype=np.float64)
    mean_delta_total = np.empty(n_chi, dtype=np.float64)
    mean_delta_high = np.empty(n_chi, dtype=np.float64)
    mean_delta_small = np.empty(n_chi, dtype=np.float64)
    mean_delta2_small = np.empty(n_chi, dtype=np.float64)
    inv_cdf = np.empty((n_chi, n_q), dtype=np.float64)

    n_u = 1025
    g23 = float(gamma_func(2.0 / 3.0))

    for i, chi_val in enumerate(chi_grid):
        delta_max = (1.5 * _Y_TILDE_CUTOFF * chi_val) / (1.0 + 1.5 * _Y_TILDE_CUTOFF * chi_val)
        u_max = delta_max ** (1.0 / 3.0)
        u = np.linspace(0.0, u_max, n_u, dtype=np.float64)
        delta = u**3

        integrand = np.zeros(n_u, dtype=np.float64)
        # Analytical u -> 0 limit of 3 u^2 * S(chi, u^3)
        integrand[0] = 3.0 * g23 * ((3.0 * chi_val) ** (2.0 / 3.0))

        pos = slice(1, n_u)
        d_pos = delta[pos]
        one_minus_d = 1.0 - d_pos
        y_tilde = (2.0 / (3.0 * chi_val)) * (d_pos / one_minus_d)
        valid = y_tilde <= _Y_TILDE_CUTOFF
        if np.any(valid):
            d_v = d_pos[valid]
            omd_v = one_minus_d[valid]
            y_v = y_tilde[valid]
            u_v = u[1:][valid]
            s_v = (omd_v + 1.0 / omd_v) * kv(2.0 / 3.0, y_v) - _eval_k13_tail(y_v)
            s_v = np.maximum(0.0, s_v)
            integrand[1:][valid] = 3.0 * (u_v**2) * s_v

        cdf = np.empty(n_u, dtype=np.float64)
        cdf[0] = 0.0
        cdf[1:] = cumulative_trapezoid(integrand, u)
        g_tot = float(cdf[-1])
        g_rate[i] = g_tot

        if g_tot <= 0.0:
            high_energy_fraction[i] = 0.0
            loss_gt_1pct_fraction[i] = 0.0
            mean_delta_total[i] = 0.0
            mean_delta_high[i] = 0.0
            mean_delta_small[i] = 0.0
            mean_delta2_small[i] = 0.0
            inv_cdf[i, :] = 0.0
            continue

        # Compute survival function directly from the right end to avoid 1 - cdf_norm cancellation
        sf = np.empty(n_u, dtype=np.float64)
        sf[-1] = 0.0
        sf[:-1] = -cumulative_trapezoid(integrand[::-1], u[::-1])[::-1]
        sf_norm = np.maximum(sf / g_tot, 1.0e-300)
        neg_log_sf = -np.log(sf_norm)
        neg_log_sf[0] = 0.0
        neg_log_sf = np.maximum.accumulate(neg_log_sf)
        inv_cdf[i, :] = np.interp(t_grid, neg_log_sf, delta)
        inv_cdf[i, 0] = 0.0

        # Cumulative mean delta = int_0^u delta(u') * integrand(u') du' / g_tot
        e_cum = np.empty(n_u, dtype=np.float64)
        e_cum[0] = 0.0
        e_cum[1:] = cumulative_trapezoid(delta * integrand, u)
        mean_delta_total[i] = float(e_cum[-1] / g_tot)

        # Cumulative mean delta^2 = int_0^u delta(u')^2 * integrand(u') du'
        e2_cum = np.empty(n_u, dtype=np.float64)
        e2_cum[0] = 0.0
        e2_cum[1:] = cumulative_trapezoid((delta ** 2) * integrand, u)

        # Fraction above delta = 0.01
        if delta_max > 0.01:
            u_001 = 0.01 ** (1.0 / 3.0)
            cdf_001 = float(np.interp(u_001, u, cdf))
            loss_gt_1pct_fraction[i] = max(0.0, min(1.0, (g_tot - cdf_001) / g_tot))
        else:
            loss_gt_1pct_fraction[i] = 0.0

        # Small-loss moments E[delta * 1_{delta <= 0.01}] and E[delta^2 * 1_{delta <= 0.01}]
        if delta_max > 0.01:
            u_001 = 0.01 ** (1.0 / 3.0)
            e_001 = float(np.interp(u_001, u, e_cum))
            e2_001 = float(np.interp(u_001, u, e2_cum))
            mean_delta_small[i] = e_001 / g_tot
            mean_delta2_small[i] = e2_001 / g_tot
        else:
            mean_delta_small[i] = float(e_cum[-1] / g_tot)
            mean_delta2_small[i] = float(e2_cum[-1] / g_tot)

        # Fraction and conditional mean above delta = 0.5
        if delta_max > 0.5:
            u_05 = 0.5 ** (1.0 / 3.0)
            cdf_05 = float(np.interp(u_05, u, cdf))
            e_05 = float(np.interp(u_05, u, e_cum))
            g_high = max(0.0, g_tot - cdf_05)
            high_energy_fraction[i] = max(0.0, min(1.0, g_high / g_tot))
            mean_delta_high[i] = float((e_cum[-1] - e_05) / g_high) if g_high > 0.0 else 0.0
        else:
            high_energy_fraction[i] = 0.0
            mean_delta_high[i] = 0.0

    return LCFATable(
        chi_grid=chi_grid,
        log_chi_grid=log_chi_grid,
        q_grid=q_grid,
        g_rate=g_rate,
        high_energy_fraction=high_energy_fraction,
        loss_gt_1pct_fraction=loss_gt_1pct_fraction,
        mean_delta_total=mean_delta_total,
        mean_delta_high=mean_delta_high,
        mean_delta_small=mean_delta_small,
        mean_delta2_small=mean_delta2_small,
        inv_cdf=inv_cdf,
    )


_DEFAULT_LCFA_TABLE: LCFATable | None = None


def _get_default_lcfa_table() -> LCFATable:
    global _DEFAULT_LCFA_TABLE
    if _DEFAULT_LCFA_TABLE is None:
        _DEFAULT_LCFA_TABLE = build_lcfa_table()
    return _DEFAULT_LCFA_TABLE


@njit(cache=True)
def _interp_table_1d_numba(chi: float, log_chi_grid: np.ndarray, values: np.ndarray) -> float:
    n = log_chi_grid.shape[0]
    chi_min = math.exp(log_chi_grid[0])
    if chi <= 0.0:
        return 0.0
    if chi <= chi_min:
        return values[0] * (chi / chi_min)
    lc = math.log(chi)
    if lc >= log_chi_grid[n - 1]:
        return values[n - 1]
    d_lc = (log_chi_grid[n - 1] - log_chi_grid[0]) / (n - 1)
    u = (lc - log_chi_grid[0]) / d_lc
    idx = int(math.floor(u))
    if idx >= n - 1:
        idx = n - 2
    w = u - idx
    return (1.0 - w) * values[idx] + w * values[idx + 1]


@njit(cache=True)
def _lookup_g_rate_numba(chi: float, log_chi_grid: np.ndarray, g_rate: np.ndarray) -> float:
    if chi <= 0.0:
        return 0.0
    chi_min = math.exp(log_chi_grid[0])
    if chi < chi_min:
        return _ASymptote_G_COEFF * chi
    lc = math.log(chi)
    n = log_chi_grid.shape[0]
    if lc >= log_chi_grid[n - 1]:
        # Extrapolate total emission rate G(chi) with chi^(2/3) scaling above chi_max
        return g_rate[n - 1] * math.exp((2.0 / 3.0) * (lc - log_chi_grid[n - 1]))
    return _interp_table_1d_numba(chi, log_chi_grid, g_rate)


@njit(cache=True)
def _lookup_inv_cdf_numba(
    chi: float,
    q: float,
    log_chi_grid: np.ndarray,
    inv_cdf: np.ndarray,
) -> float:
    n_chi = log_chi_grid.shape[0]
    n_q = inv_cdf.shape[1]
    if chi <= 0.0 or q <= 0.0:
        return 0.0
    if q >= 1.0:
        q = 1.0 - 1.0e-15

    lc = math.log(chi)
    if lc <= log_chi_grid[0]:
        ic0 = 0
        ic1 = 0
        wc = 0.0
    elif lc >= log_chi_grid[n_chi - 1]:
        ic0 = n_chi - 1
        ic1 = n_chi - 1
        wc = 0.0
    else:
        d_lc = (log_chi_grid[n_chi - 1] - log_chi_grid[0]) / (n_chi - 1)
        uc = (lc - log_chi_grid[0]) / d_lc
        ic0 = int(math.floor(uc))
        if ic0 >= n_chi - 1:
            ic0 = n_chi - 2
        ic1 = ic0 + 1
        wc = uc - ic0

    t = -math.log(max(1.0 - q, 1.0e-15))
    s = math.sqrt(min(t, _INV_CDF_T_MAX) / _INV_CDF_T_MAX)
    uq = s * (n_q - 1)
    iq0 = int(math.floor(uq))
    if iq0 >= n_q - 1:
        iq0 = n_q - 2
    iq1 = iq0 + 1
    wq = uq - iq0

    v00 = inv_cdf[ic0, iq0]
    v01 = inv_cdf[ic0, iq1]
    v10 = inv_cdf[ic1, iq0]
    v11 = inv_cdf[ic1, iq1]

    v0 = (1.0 - wq) * v00 + wq * v01
    v1 = (1.0 - wq) * v10 + wq * v11
    return (1.0 - wc) * v0 + wc * v1


def lcfa_rate_per_s(
    gamma: float | np.ndarray,
    chi: float | np.ndarray,
    table: LCFATable | None = None,
) -> float | np.ndarray:
    """Evaluate total LCFA synchrotron emission rate W_rad(gamma, chi) in s^-1."""
    tbl = _get_default_lcfa_table() if table is None else table
    g_arr = np.asarray(gamma, dtype=np.float64)
    c_arr = np.asarray(chi, dtype=np.float64)
    if not np.all(np.isfinite(g_arr)) or np.any(g_arr <= 1.0):
        raise ValueError("gamma must be finite and > 1.0")
    if not np.all(np.isfinite(c_arr)) or np.any(c_arr < 0.0):
        raise ValueError("chi must be finite and >= 0.0")

    gb, cb = np.broadcast_arrays(g_arr, c_arr)
    flat_g = gb.ravel()
    flat_c = cb.ravel()
    out = np.empty_like(flat_g, dtype=np.float64)
    for k in range(flat_g.size):
        g_val = _lookup_g_rate_numba(float(flat_c[k]), tbl.log_chi_grid, tbl.g_rate)
        out[k] = (W0_LCFA_PER_S / float(flat_g[k])) * g_val

    if np.ndim(gamma) == 0 and np.ndim(chi) == 0:
        return float(out[0])
    return out.reshape(gb.shape)


@njit(cache=True)
def _simulate_mc_numba(
    r0_m: np.ndarray,
    p0_kgms: np.ndarray,
    b_impact_m: np.ndarray,
    dt_jitter_s: np.ndarray,
    gamma0: float,
    t_s: np.ndarray,
    dt_s: float,
    use_vay: bool,
    Ec: np.ndarray,
    Bc: np.ndarray,
    freqs: np.ndarray,
    x0_m: float,
    dx_m: float,
    Nx: int,
    y0_m: float,
    dy_m: float,
    Ny: int,
    z0_m: float,
    dz_m: float,
    Nz: int,
    use_gauss_env: bool,
    sigma_t_s: float,
    t_ref_s: float,
    log_chi_grid: np.ndarray,
    g_rate: np.ndarray,
    inv_cdf: np.ndarray,
    u_emit: np.ndarray,
    u_delta: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_particles = r0_m.shape[0]
    n_steps = t_s.shape[0]

    r_trace = np.empty((n_particles, n_steps, 3), dtype=np.float64)
    p_trace = np.empty((n_particles, n_steps, 3), dtype=np.float64)
    gamma_trace = np.empty((n_particles, n_steps), dtype=np.float64)
    chi_trace = np.empty((n_particles, n_steps), dtype=np.float64)
    alpha_trace = np.empty((n_particles, n_steps), dtype=np.float64)
    p_no_loss_trace = np.ones((n_particles, n_steps), dtype=np.float64)

    # First pass count or preallocate dynamic capacity
    cap = max(256, n_particles * 4)
    ph_energy = np.empty(cap, dtype=np.float64)
    ph_delta_init = np.empty(cap, dtype=np.float64)
    ph_alpha = np.empty(cap, dtype=np.float64)
    ph_chi = np.empty(cap, dtype=np.float64)
    ph_gamma_prior = np.empty(cap, dtype=np.float64)
    ph_gamma0 = np.empty(cap, dtype=np.float64)
    ph_b = np.empty(cap, dtype=np.float64)
    ph_weight = np.empty(cap, dtype=np.float64)
    ph_t = np.empty(cap, dtype=np.float64)
    ph_idx = np.empty(cap, dtype=np.int64)
    n_ph = 0

    for i in range(n_particles):
        r = r0_m[i].copy()
        p = p0_kgms[i].copy()
        jitter = dt_jitter_s[i]
        b_imp = b_impact_m[i]
        lost_gt_1pct = False

        for n in range(n_steps):
            t_curr = t_s[n]
            e_vm, b_t = evaluate_eb_numba(
                r[0],
                r[1],
                r[2],
                t_curr,
                jitter,
                Ec,
                Bc,
                freqs,
                x0_m,
                dx_m,
                Nx,
                y0_m,
                dy_m,
                Ny,
                z0_m,
                dz_m,
                Nz,
                use_gauss_env,
                sigma_t_s,
                t_ref_s,
            )

            ux = p[0] / MC_KGMS
            uy = p[1] / MC_KGMS
            uz = p[2] / MC_KGMS
            gamma_curr = math.sqrt(1.0 + ux * ux + uy * uy + uz * uz)
            chi_curr = compute_chi_numba(e_vm, b_t, p)
            alpha_curr = compute_deviation_angle_numba(p, signed_z=False)

            r_trace[i, n, 0] = r[0]
            r_trace[i, n, 1] = r[1]
            r_trace[i, n, 2] = r[2]
            p_trace[i, n, 0] = p[0]
            p_trace[i, n, 1] = p[1]
            p_trace[i, n, 2] = p[2]
            gamma_trace[i, n] = gamma_curr
            chi_trace[i, n] = chi_curr
            alpha_trace[i, n] = alpha_curr
            p_no_loss_trace[i, n] = 0.0 if lost_gt_1pct else 1.0

            if n < n_steps - 1:
                if use_vay:
                    r_next, p_prior = step_vay_numba(r, p, e_vm, b_t, dt_s)
                else:
                    r_next, p_prior = step_boris_numba(r, p, e_vm, b_t, dt_s)

                ux_p = p_prior[0] / MC_KGMS
                uy_p = p_prior[1] / MC_KGMS
                uz_p = p_prior[2] / MC_KGMS
                gamma_prior = math.sqrt(1.0 + ux_p * ux_p + uy_p * uy_p + uz_p * uz_p)
                chi_emit = compute_chi_numba(e_vm, b_t, p_prior)

                if chi_emit > 0.0 and gamma_prior > 1.0:
                    w_rad = (W0_LCFA_PER_S / gamma_prior) * _lookup_g_rate_numba(
                        chi_emit, log_chi_grid, g_rate
                    )
                    p_emit = -math.expm1(-w_rad * dt_s)
                    if u_emit[i, n] < p_emit:
                        delta = _lookup_inv_cdf_numba(chi_emit, u_delta[i, n], log_chi_grid, inv_cdf)
                        if delta > 0.0:
                            # Grow arrays if needed
                            if n_ph >= cap:
                                new_cap = cap * 2
                                tmp_e = np.empty(new_cap, dtype=np.float64)
                                tmp_di = np.empty(new_cap, dtype=np.float64)
                                tmp_a = np.empty(new_cap, dtype=np.float64)
                                tmp_c = np.empty(new_cap, dtype=np.float64)
                                tmp_gp = np.empty(new_cap, dtype=np.float64)
                                tmp_g0 = np.empty(new_cap, dtype=np.float64)
                                tmp_b = np.empty(new_cap, dtype=np.float64)
                                tmp_w = np.empty(new_cap, dtype=np.float64)
                                tmp_t = np.empty(new_cap, dtype=np.float64)
                                tmp_i = np.empty(new_cap, dtype=np.int64)
                                for m in range(cap):
                                    tmp_e[m] = ph_energy[m]
                                    tmp_di[m] = ph_delta_init[m]
                                    tmp_a[m] = ph_alpha[m]
                                    tmp_c[m] = ph_chi[m]
                                    tmp_gp[m] = ph_gamma_prior[m]
                                    tmp_g0[m] = ph_gamma0[m]
                                    tmp_b[m] = ph_b[m]
                                    tmp_w[m] = ph_weight[m]
                                    tmp_t[m] = ph_t[m]
                                    tmp_i[m] = ph_idx[m]
                                ph_energy = tmp_e
                                ph_delta_init = tmp_di
                                ph_alpha = tmp_a
                                ph_chi = tmp_c
                                ph_gamma_prior = tmp_gp
                                ph_gamma0 = tmp_g0
                                ph_b = tmp_b
                                ph_weight = tmp_w
                                ph_t = tmp_t
                                ph_idx = tmp_i
                                cap = new_cap

                            ph_energy[n_ph] = delta * gamma_prior * MC2_GEV
                            ph_delta_init[n_ph] = (delta * gamma_prior) / gamma0
                            ph_alpha[n_ph] = compute_deviation_angle_numba(p_prior, signed_z=False)
                            ph_chi[n_ph] = chi_emit
                            ph_gamma_prior[n_ph] = gamma_prior
                            ph_gamma0[n_ph] = gamma0
                            ph_b[n_ph] = b_imp
                            ph_weight[n_ph] = 1.0
                            ph_t[n_ph] = t_curr
                            ph_idx[n_ph] = i
                            n_ph += 1

                            recoil = max(0.0, 1.0 - delta)
                            p_prior[0] *= recoil
                            p_prior[1] *= recoil
                            p_prior[2] *= recoil
                            ux_n = p_prior[0] / MC_KGMS
                            uy_n = p_prior[1] / MC_KGMS
                            uz_n = p_prior[2] / MC_KGMS
                            if math.sqrt(1.0 + ux_n * ux_n + uy_n * uy_n + uz_n * uz_n) < 0.99 * gamma0:
                                lost_gt_1pct = True

                r = r_next
                p = p_prior

    return (
        r_trace,
        p_trace,
        gamma_trace,
        chi_trace,
        alpha_trace,
        p_no_loss_trace,
        ph_energy[:n_ph],
        ph_delta_init[:n_ph],
        ph_alpha[:n_ph],
        ph_chi[:n_ph],
        ph_gamma_prior[:n_ph],
        ph_gamma0[:n_ph],
        ph_b[:n_ph],
        ph_weight[:n_ph],
        ph_t[:n_ph],
        ph_idx[:n_ph],
    )


@njit(cache=True)
def _lookup_cdf_tail_from_invcdf_numba(
    chi: float,
    delta_cut: float,
    log_chi_grid: np.ndarray,
    inv_cdf: np.ndarray,
) -> float:
    """Return P(delta > delta_cut | chi) by inverting the monotonic inv_cdf(chi, q) table."""
    if chi <= 0.0 or delta_cut >= 1.0:
        return 0.0
    if delta_cut <= 0.0:
        return 1.0
    q_max = -math.expm1(-_INV_CDF_T_MAX)
    if _lookup_inv_cdf_numba(chi, q_max, log_chi_grid, inv_cdf) <= delta_cut:
        return 0.0
    lo = 0.0
    hi = 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        t_mid = mid * mid * _INV_CDF_T_MAX
        q_mid = -math.expm1(-t_mid)
        if _lookup_inv_cdf_numba(chi, q_mid, log_chi_grid, inv_cdf) < delta_cut:
            lo = mid
        else:
            hi = mid
    s_mid = 0.5 * (lo + hi)
    return math.exp(-(s_mid * s_mid) * _INV_CDF_T_MAX)


@njit(cache=True)
def _p_small_loss_ok_numba(mean_small: float, var_small: float) -> float:
    """Clamped Gaussian survival P(sum of small losses, each delta <= 0.01, stays < 0.01)."""
    if var_small <= 0.0:
        return 1.0 if mean_small < 0.01 else 0.0
    return 0.5 * math.erfc((mean_small - 0.01) / math.sqrt(2.0 * var_small))


@njit(cache=True)
def _simulate_weighted_numba(
    r0_m: np.ndarray,
    p0_kgms: np.ndarray,
    b_impact_m: np.ndarray,
    dt_jitter_s: np.ndarray,
    gamma0: float,
    t_s: np.ndarray,
    dt_s: float,
    use_vay: bool,
    Ec: np.ndarray,
    Bc: np.ndarray,
    freqs: np.ndarray,
    x0_m: float,
    dx_m: float,
    Nx: int,
    y0_m: float,
    dy_m: float,
    Ny: int,
    z0_m: float,
    dz_m: float,
    Nz: int,
    use_gauss_env: bool,
    sigma_t_s: float,
    t_ref_s: float,
    log_chi_grid: np.ndarray,
    g_rate: np.ndarray,
    inv_cdf: np.ndarray,
    high_energy_fraction: np.ndarray,
    loss_gt_1pct_fraction: np.ndarray,
    mean_delta_total: np.ndarray,
    mean_delta_high: np.ndarray,
    mean_delta_small: np.ndarray,
    mean_delta2_small: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_particles = r0_m.shape[0]
    n_steps = t_s.shape[0]

    r_trace = np.empty((n_particles, n_steps, 3), dtype=np.float64)
    p_trace = np.empty((n_particles, n_steps, 3), dtype=np.float64)
    gamma_trace = np.empty((n_particles, n_steps), dtype=np.float64)
    chi_trace = np.empty((n_particles, n_steps), dtype=np.float64)
    alpha_trace = np.empty((n_particles, n_steps), dtype=np.float64)
    p_no_loss_trace = np.empty((n_particles, n_steps), dtype=np.float64)

    # Up to 2 weighted photon records (fresh + depleted branch) per step per particle
    max_ph = n_particles * max(1, n_steps - 1) * 2
    ph_energy = np.empty(max_ph, dtype=np.float64)
    ph_delta_init = np.empty(max_ph, dtype=np.float64)
    ph_alpha = np.empty(max_ph, dtype=np.float64)
    ph_chi = np.empty(max_ph, dtype=np.float64)
    ph_gamma_prior = np.empty(max_ph, dtype=np.float64)
    ph_gamma0 = np.empty(max_ph, dtype=np.float64)
    ph_b = np.empty(max_ph, dtype=np.float64)
    ph_weight = np.empty(max_ph, dtype=np.float64)
    ph_t = np.empty(max_ph, dtype=np.float64)
    ph_idx = np.empty(max_ph, dtype=np.int64)
    n_ph = 0

    for i in range(n_particles):
        r = r0_m[i].copy()
        p = p0_kgms[i].copy()
        jitter = dt_jitter_s[i]
        b_imp = b_impact_m[i]
        p_no_loss = 1.0
        p_no_big = 1.0
        cum_small_mean = 0.0
        cum_small_var = 0.0
        gamma_avg = gamma0

        for n in range(n_steps):
            t_curr = t_s[n]
            e_vm, b_t = evaluate_eb_numba(
                r[0],
                r[1],
                r[2],
                t_curr,
                jitter,
                Ec,
                Bc,
                freqs,
                x0_m,
                dx_m,
                Nx,
                y0_m,
                dy_m,
                Ny,
                z0_m,
                dz_m,
                Nz,
                use_gauss_env,
                sigma_t_s,
                t_ref_s,
            )

            chi_curr = compute_chi_numba(e_vm, b_t, p)
            alpha_curr = compute_deviation_angle_numba(p, signed_z=False)

            r_trace[i, n, 0] = r[0]
            r_trace[i, n, 1] = r[1]
            r_trace[i, n, 2] = r[2]
            p_trace[i, n, 0] = p[0]
            p_trace[i, n, 1] = p[1]
            p_trace[i, n, 2] = p[2]
            gamma_trace[i, n] = gamma_avg
            chi_trace[i, n] = chi_curr
            alpha_trace[i, n] = alpha_curr
            p_no_loss_trace[i, n] = p_no_loss

            if n < n_steps - 1:
                if use_vay:
                    r_next, p_next = step_vay_numba(r, p, e_vm, b_t, dt_s)
                else:
                    r_next, p_next = step_boris_numba(r, p, e_vm, b_t, dt_s)

                if chi_curr > 0.0:
                    g_val = _lookup_g_rate_numba(chi_curr, log_chi_grid, g_rate)
                    w_rad_0 = (W0_LCFA_PER_S / gamma0) * g_val
                    f_high_0 = _interp_table_1d_numba(chi_curr, log_chi_grid, high_energy_fraction)
                    f_1pct = _interp_table_1d_numba(chi_curr, log_chi_grid, loss_gt_1pct_fraction)
                    mean_dh_0 = _interp_table_1d_numba(chi_curr, log_chi_grid, mean_delta_high)
                    mean_dtot_0 = _interp_table_1d_numba(chi_curr, log_chi_grid, mean_delta_total)

                    w_loss_1pct = w_rad_0 * f_1pct
                    mean_ds = _interp_table_1d_numba(chi_curr, log_chi_grid, mean_delta_small)
                    mean_d2s = _interp_table_1d_numba(chi_curr, log_chi_grid, mean_delta2_small)

                    # Mid-step no-loss survival: no single delta > 0.01 emission (p_no_big_mid)
                    # and accumulated small losses (each delta <= 0.01) still below 0.01 in total
                    p_no_big_mid = p_no_big * math.exp(-0.5 * w_loss_1pct * dt_s)
                    m_mid = cum_small_mean + 0.5 * w_rad_0 * mean_ds * dt_s
                    v_mid = cum_small_var + 0.5 * w_rad_0 * mean_d2s * dt_s
                    p_small_ok_mid = _p_small_loss_ok_numba(m_mid, v_mid)
                    p_no_loss_mid = max(0.0, min(p_no_loss, p_no_big_mid * p_small_ok_mid))

                    # Fresh (non-depleted) branch: gamma_prior = gamma0 >= 0.99 * gamma0
                    dw_fresh = p_no_loss_mid * w_rad_0 * f_high_0 * dt_s
                    if dw_fresh > 0.0 and mean_dh_0 > 0.5:
                        ph_energy[n_ph] = mean_dh_0 * gamma0 * MC2_GEV
                        ph_delta_init[n_ph] = mean_dh_0
                        ph_alpha[n_ph] = alpha_curr
                        ph_chi[n_ph] = chi_curr
                        ph_gamma_prior[n_ph] = gamma0
                        ph_gamma0[n_ph] = gamma0
                        ph_b[n_ph] = b_imp
                        ph_weight[n_ph] = dw_fresh
                        ph_t[n_ph] = t_curr
                        ph_idx[n_ph] = i
                        n_ph += 1

                    # Depleted branch (< 0.99 gamma0) contributing background to N_total if gamma_dep > 0.5 gamma0
                    p_dep = max(0.0, 1.0 - p_no_loss_mid)
                    if p_dep > 1.0e-12:
                        gamma_dep = (gamma_avg - p_no_loss_mid * gamma0) / p_dep
                        gamma_dep = max(1.0, min(gamma_dep, 0.985 * gamma0))
                        if gamma_dep > 0.5 * gamma0:
                            chi_dep = chi_curr * (gamma_dep / gamma0)
                            g_dep = _lookup_g_rate_numba(chi_dep, log_chi_grid, g_rate)
                            w_rad_dep = (W0_LCFA_PER_S / gamma_dep) * g_dep
                            delta_cut = 0.5 * (gamma0 / gamma_dep)
                            f_above_cut = _lookup_cdf_tail_from_invcdf_numba(
                                chi_dep, delta_cut, log_chi_grid, inv_cdf
                            )
                            dw_dep = p_dep * w_rad_dep * f_above_cut * dt_s
                            if dw_dep > 0.0:
                                delta_rep = 0.5 * (delta_cut + 1.0)
                                delta_init_dep = delta_rep * (gamma_dep / gamma0)
                                ph_energy[n_ph] = delta_init_dep * gamma0 * MC2_GEV
                                ph_delta_init[n_ph] = delta_init_dep
                                ph_alpha[n_ph] = alpha_curr * (gamma0 / gamma_dep)
                                ph_chi[n_ph] = chi_dep
                                ph_gamma_prior[n_ph] = gamma_dep
                                ph_gamma0[n_ph] = gamma0
                                ph_b[n_ph] = b_imp
                                ph_weight[n_ph] = dw_dep
                                ph_t[n_ph] = t_curr
                                ph_idx[n_ph] = i
                                n_ph += 1

                    p_no_big = p_no_big * math.exp(-w_loss_1pct * dt_s)
                    cum_small_mean = cum_small_mean + w_rad_0 * mean_ds * dt_s
                    cum_small_var = cum_small_var + w_rad_0 * mean_d2s * dt_s
                    p_small_ok_end = _p_small_loss_ok_numba(cum_small_mean, cum_small_var)
                    p_no_loss = max(0.0, min(p_no_loss, p_no_big * p_small_ok_end))
                    if gamma_avg > 1.0:
                        chi_avg = chi_curr * (gamma_avg / gamma0)
                        g_avg = _lookup_g_rate_numba(chi_avg, log_chi_grid, g_rate)
                        mean_dtot_avg = _interp_table_1d_numba(chi_avg, log_chi_grid, mean_delta_total)
                        dgamma = (W0_LCFA_PER_S * g_avg * mean_dtot_avg) * dt_s
                        gamma_avg = max(1.0, gamma_avg - dgamma)

                r = r_next
                p = p_next

    return (
        r_trace,
        p_trace,
        gamma_trace,
        chi_trace,
        alpha_trace,
        p_no_loss_trace,
        ph_energy[:n_ph],
        ph_delta_init[:n_ph],
        ph_alpha[:n_ph],
        ph_chi[:n_ph],
        ph_gamma_prior[:n_ph],
        ph_gamma0[:n_ph],
        ph_b[:n_ph],
        ph_weight[:n_ph],
        ph_t[:n_ph],
        ph_idx[:n_ph],
    )


def simulate_qed_ensemble(
    pack: FieldPack,
    batch: ElectronBatch,
    dt_s: float | None = None,
    t_start_s: float | None = None,
    t_end_s: float | None = None,
    mode: str = "monte_carlo",
    method: str = "vay",
    table: LCFATable | None = None,
    rng: np.random.Generator | None = None,
    chi_max_ref: float | None = None,
) -> QEDResult:
    """Propagate an ElectronBatch through a FieldPack with stochastic or weighted LCFA QED."""
    if chi_max_ref is not None and (not math.isfinite(chi_max_ref) or chi_max_ref <= 0.0):
        raise ValueError(f"chi_max_ref must be finite and positive when provided, got {chi_max_ref}")
    if mode not in ("monte_carlo", "weighted"):
        raise ValueError(f"mode must be 'monte_carlo' or 'weighted', got {mode!r}")
    if method not in ("vay", "boris"):
        raise ValueError(f"method must be 'vay' or 'boris', got {method!r}")

    t_s, dt_eff = build_simulation_time_grid(
        pack, batch, dt_s=dt_s, t_start_s=t_start_s, t_end_s=t_end_s
    )
    n_steps = t_s.shape[0]

    tbl = _get_default_lcfa_table() if table is None else table
    use_gauss_env = (pack.freqs.size == 1) and (pack.pulse.model == "gauss")
    sigma_t = pack.pulse.sigma_t_s if use_gauss_env else 1.0

    r0_arr = np.ascontiguousarray(batch.r0_m, dtype=np.float64)
    p0_arr = np.ascontiguousarray(batch.p0_kgms, dtype=np.float64)
    b_arr = np.ascontiguousarray(batch.b_impact_m, dtype=np.float64)
    jitter_arr = np.ascontiguousarray(batch.dt_jitter_s, dtype=np.float64)

    if mode == "monte_carlo":
        gen = np.random.default_rng() if rng is None else rng
        u_emit = gen.uniform(0.0, 1.0, size=(r0_arr.shape[0], n_steps - 1))
        u_delta = gen.uniform(0.0, 1.0, size=(r0_arr.shape[0], n_steps - 1))
        res = _simulate_mc_numba(
            r0_arr,
            p0_arr,
            b_arr,
            jitter_arr,
            batch.gamma0,
            t_s,
            dt_eff,
            method == "vay",
            pack.Ec,
            pack.Bc,
            pack.freqs,
            pack.grid.x0_m,
            pack.grid.dx_m,
            pack.grid.Nx,
            pack.grid.y0_m,
            pack.grid.dy_m,
            pack.grid.Ny,
            pack.grid.z0_m,
            pack.grid.dz_m,
            pack.grid.Nz,
            use_gauss_env,
            sigma_t,
            pack.pulse.t_ref_s,
            tbl.log_chi_grid,
            tbl.g_rate,
            tbl.inv_cdf,
            u_emit,
            u_delta,
        )
    else:
        res = _simulate_weighted_numba(
            r0_arr,
            p0_arr,
            b_arr,
            jitter_arr,
            batch.gamma0,
            t_s,
            dt_eff,
            method == "vay",
            pack.Ec,
            pack.Bc,
            pack.freqs,
            pack.grid.x0_m,
            pack.grid.dx_m,
            pack.grid.Nx,
            pack.grid.y0_m,
            pack.grid.dy_m,
            pack.grid.Ny,
            pack.grid.z0_m,
            pack.grid.dz_m,
            pack.grid.Nz,
            use_gauss_env,
            sigma_t,
            pack.pulse.t_ref_s,
            tbl.log_chi_grid,
            tbl.g_rate,
            tbl.inv_cdf,
            tbl.high_energy_fraction,
            tbl.loss_gt_1pct_fraction,
            tbl.mean_delta_total,
            tbl.mean_delta_high,
            tbl.mean_delta_small,
            tbl.mean_delta2_small,
        )

    (
        r_trace,
        p_trace,
        gamma_trace,
        chi_trace,
        alpha_trace,
        p_no_loss_trace,
        ph_energy,
        ph_delta_init,
        ph_alpha,
        ph_chi,
        ph_gamma_prior,
        ph_gamma0,
        ph_b,
        ph_weight,
        ph_t,
        ph_idx,
    ) = res

    catalog = PhotonCatalog(
        energy_gev=ph_energy,
        delta_init=ph_delta_init,
        alpha_rad=ph_alpha,
        chi_emit=ph_chi,
        gamma_prior=ph_gamma_prior,
        gamma0=ph_gamma0,
        b_impact_m=ph_b,
        weight=ph_weight,
        t_emit_s=ph_t,
        particle_idx=ph_idx,
    )

    return QEDResult(
        t_s=t_s,
        r_m=r_trace,
        p_kgms=p_trace,
        gamma=gamma_trace,
        chi=chi_trace,
        alpha=alpha_trace,
        p_no_loss=p_no_loss_trace,
        photons=catalog,
        batch=batch,
        mode=mode,
    )


def apply_selection_cuts(
    catalog: PhotonCatalog,
    pol: str,
    a_max: float,
    chi_max: float,
    gamma0: float | None = None,
) -> dict[str, np.ndarray]:
    """Apply Sec. V photon selection cuts for N_total and N_signal.

    Note: for pol='circular', a_max should be a_max^CP (= a_max^LP / sqrt(2)), matching FieldChecks.a_max.
    """
    if pol not in ("z-linear", "circular"):
        raise ValueError(f"pol must be 'z-linear' or 'circular', got {pol!r}")
    if not math.isfinite(a_max) or a_max <= 0.0:
        raise ValueError(f"a_max must be finite and positive, got {a_max}")
    if not math.isfinite(chi_max) or chi_max <= 0.0:
        raise ValueError(f"chi_max must be finite and positive, got {chi_max}")
    if gamma0 is not None and (not math.isfinite(gamma0) or gamma0 <= 1.0):
        raise ValueError(f"gamma0 must be finite and > 1.0, got {gamma0}")

    catalog_arrays = (
        catalog.energy_gev,
        catalog.delta_init,
        catalog.alpha_rad,
        catalog.chi_emit,
        catalog.gamma_prior,
        catalog.gamma0,
        catalog.b_impact_m,
        catalog.weight,
        catalog.t_emit_s,
        catalog.particle_idx,
    )
    n_ph = catalog.delta_init.shape[0]
    for arr in catalog_arrays:
        if arr.ndim != 1 or arr.shape[0] != n_ph:
            raise ValueError("All PhotonCatalog arrays must be 1D with matching length")
    if n_ph == 0:
        empty = np.zeros(0, dtype=bool)
        return {"total_mask": empty, "signal_mask": empty}

    gamma0_arr = (
        catalog.gamma0 if gamma0 is None else np.full(n_ph, float(gamma0), dtype=np.float64)
    )
    angle_factor = 0.6 if pol == "z-linear" else 0.7
    alpha_thresh = angle_factor * (a_max / gamma0_arr)

    total_mask = (catalog.delta_init > 0.5) & (catalog.alpha_rad > alpha_thresh)
    signal_mask = (
        total_mask
        & (catalog.chi_emit > 0.5 * chi_max)
        & (catalog.gamma_prior >= 0.99 * gamma0_arr)
    )
    return {
        "total_mask": total_mask,
        "signal_mask": signal_mask,
    }


def compute_sigma_and_eta(
    catalog: PhotonCatalog,
    pol: str,
    a_max: float,
    chi_max: float,
    b_max_m: float,
    n_electrons: int,
    gamma0: float | None = None,
    t_inj_over_tau_l: float = 1.0,
) -> dict[str, float]:
    """Compute effective QED signal cross section sigma (m^2 and cm^2) and signal ratio eta."""
    if not math.isfinite(b_max_m) or b_max_m <= 0.0:
        raise ValueError(f"b_max_m must be finite and positive, got {b_max_m}")
    if not isinstance(n_electrons, (int, np.integer)) or n_electrons < 1:
        raise ValueError(f"n_electrons must be integer >= 1, got {n_electrons}")
    if not math.isfinite(t_inj_over_tau_l) or t_inj_over_tau_l <= 0.0:
        raise ValueError(f"t_inj_over_tau_l must be finite and positive, got {t_inj_over_tau_l}")

    masks = apply_selection_cuts(
        catalog=catalog,
        pol=pol,
        a_max=a_max,
        chi_max=chi_max,
        gamma0=gamma0,
    )
    total_mask = masks["total_mask"]
    signal_mask = masks["signal_mask"]

    w_total = float(np.sum(catalog.weight[total_mask])) if np.any(total_mask) else 0.0
    w_signal = float(np.sum(catalog.weight[signal_mask])) if np.any(signal_mask) else 0.0
    eta = (w_signal / w_total) if w_total > 0.0 else 0.0

    area_m2 = math.pi * (b_max_m**2)
    sigma_m2 = area_m2 * t_inj_over_tau_l * (w_signal / float(n_electrons))
    sigma_cm2 = 1.0e4 * sigma_m2

    return {
        "w_total": w_total,
        "w_signal": w_signal,
        "eta": float(eta),
        "sigma_m2": float(sigma_m2),
        "sigma_cm2": float(sigma_cm2),
    }
