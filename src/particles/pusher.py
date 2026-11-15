"""Relativistic Vay and Boris trajectory pushers and initial electron ensemble sampler."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from .chi import compute_chi_numba, compute_deviation_angle_numba
from .constants import C_LIGHT, E_CHARGE, MC_KGMS, M_ELECTRON, energy_gev_to_gamma, period_from_lambda
from .fields import FieldPack, evaluate_eb_numba

_Q_ELECTRON: float = -E_CHARGE


@dataclass(frozen=True)
class ElectronBatch:
    """Initial phase-space and timing-jitter ensemble for N electrons."""

    r0_m: np.ndarray  # (N, 3) float64
    p0_kgms: np.ndarray  # (N, 3) float64
    b_impact_m: np.ndarray  # (N,) float64
    phi_rad: np.ndarray  # (N,) float64
    dt_jitter_s: np.ndarray  # (N,) float64
    gamma0: float
    energy_gev: float
    b_max_m: float


@dataclass(frozen=True)
class TrajectoryResult:
    """Recorded trajectory, momentum, Lorentz factor, chi, and deviation angle traces."""

    t_s: np.ndarray  # (N_steps,) float64
    r_m: np.ndarray  # (N_particles, N_steps, 3) float64
    p_kgms: np.ndarray  # (N_particles, N_steps, 3) float64
    gamma: np.ndarray  # (N_particles, N_steps) float64
    chi: np.ndarray  # (N_particles, N_steps) float64
    alpha: np.ndarray  # (N_particles, N_steps) float64
    batch: ElectronBatch


def sample_electron_batch(
    n_particles: int,
    energy_gev: float,
    b_max_m: float,
    y0_m: float,
    sigma_dt_s: float = 0.0,
    jitter_mode: str = "per_shot",
    rng: np.random.Generator | None = None,
) -> ElectronBatch:
    """Sample an electron ensemble with p(b) = 2b/b_max^2 and initial momentum along +y."""
    if not isinstance(n_particles, (int, np.integer)) or n_particles < 1:
        raise ValueError(f"n_particles must be an integer >= 1, got {n_particles}")
    gamma0 = energy_gev_to_gamma(float(energy_gev))
    if not math.isfinite(b_max_m) or b_max_m < 0.0:
        raise ValueError(f"b_max_m must be finite and non-negative, got {b_max_m}")
    if not math.isfinite(y0_m):
        raise ValueError(f"y0_m must be finite, got {y0_m}")
    if not math.isfinite(sigma_dt_s) or sigma_dt_s < 0.0:
        raise ValueError(f"sigma_dt_s must be finite and non-negative, got {sigma_dt_s}")
    if jitter_mode not in ("per_shot", "per_particle"):
        raise ValueError(f"jitter_mode must be 'per_shot' or 'per_particle', got {jitter_mode!r}")

    gen = np.random.default_rng() if rng is None else rng
    n = int(n_particles)
    u = gen.uniform(0.0, 1.0, size=n)
    phi_rad = gen.uniform(0.0, 2.0 * math.pi, size=n)
    b_impact_m = float(b_max_m) * np.sqrt(u)

    r0_m = np.empty((n, 3), dtype=np.float64)
    r0_m[:, 0] = b_impact_m * np.cos(phi_rad)
    r0_m[:, 1] = float(y0_m)
    r0_m[:, 2] = b_impact_m * np.sin(phi_rad)

    p0_y = MC_KGMS * math.sqrt(gamma0 * gamma0 - 1.0)
    p0_kgms = np.zeros((n, 3), dtype=np.float64)
    p0_kgms[:, 1] = p0_y

    if sigma_dt_s == 0.0:
        dt_jitter_s = np.zeros(n, dtype=np.float64)
    elif jitter_mode == "per_shot":
        dt_jitter_s = np.full(n, float(gen.normal(0.0, sigma_dt_s)), dtype=np.float64)
    else:
        dt_jitter_s = gen.normal(0.0, sigma_dt_s, size=n).astype(np.float64)

    return ElectronBatch(
        r0_m=r0_m,
        p0_kgms=p0_kgms,
        b_impact_m=b_impact_m,
        phi_rad=phi_rad,
        dt_jitter_s=dt_jitter_s,
        gamma0=float(gamma0),
        energy_gev=float(energy_gev),
        b_max_m=float(b_max_m),
    )


@njit(cache=True)
def step_vay_numba(
    r_m: np.ndarray,
    p_kgms: np.ndarray,
    E_vm: np.ndarray,
    B_t: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one electron (r_m, p_kgms) by dt_s using the relativistic Vay (2008) pusher."""
    ux = p_kgms[0] / MC_KGMS
    uy = p_kgms[1] / MC_KGMS
    uz = p_kgms[2] / MC_KGMS
    gamma_n = math.sqrt(1.0 + ux * ux + uy * uy + uz * uz)

    vx_c = ux / gamma_n
    vy_c = uy / gamma_n
    vz_c = uz / gamma_n

    eps_scale = (_Q_ELECTRON * dt_s) / (2.0 * MC_KGMS)
    tau_scale = (_Q_ELECTRON * dt_s) / (2.0 * M_ELECTRON)

    ex_eps = eps_scale * E_vm[0]
    ey_eps = eps_scale * E_vm[1]
    ez_eps = eps_scale * E_vm[2]

    tx = tau_scale * B_t[0]
    ty = tau_scale * B_t[1]
    tz = tau_scale * B_t[2]

    # u' = u^n + 2 * eps + (u^n / gamma^n) x tau
    upx = ux + 2.0 * ex_eps + (vy_c * tz - vz_c * ty)
    upy = uy + 2.0 * ey_eps + (vz_c * tx - vx_c * tz)
    upz = uz + 2.0 * ez_eps + (vx_c * ty - vy_c * tx)

    tau_sq = tx * tx + ty * ty + tz * tz
    up_sq = upx * upx + upy * upy + upz * upz
    u_star = upx * tx + upy * ty + upz * tz

    sigma = 1.0 + up_sq - tau_sq
    disc = sigma * sigma + 4.0 * (tau_sq + u_star * u_star)
    gamma_next = math.sqrt(0.5 * (sigma + math.sqrt(disc)))

    inv_g = 1.0 / gamma_next
    ttx = tx * inv_g
    tty = ty * inv_g
    ttz = tz * inv_g

    s = 1.0 / (1.0 + ttx * ttx + tty * tty + ttz * ttz)
    up_dot_t = upx * ttx + upy * tty + upz * ttz

    unx = s * (upx + up_dot_t * ttx + (upy * ttz - upz * tty))
    uny = s * (upy + up_dot_t * tty + (upz * ttx - upx * ttz))
    unz = s * (upz + up_dot_t * ttz + (upx * tty - upy * ttx))

    p_next = np.empty(3, dtype=np.float64)
    p_next[0] = MC_KGMS * unx
    p_next[1] = MC_KGMS * uny
    p_next[2] = MC_KGMS * unz

    v_dt = (C_LIGHT * dt_s) / gamma_next
    r_next = np.empty(3, dtype=np.float64)
    r_next[0] = r_m[0] + unx * v_dt
    r_next[1] = r_m[1] + uny * v_dt
    r_next[2] = r_m[2] + unz * v_dt

    return r_next, p_next


@njit(cache=True)
def step_boris_numba(
    r_m: np.ndarray,
    p_kgms: np.ndarray,
    E_vm: np.ndarray,
    B_t: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one electron (r_m, p_kgms) by dt_s using the relativistic Boris pusher."""
    ux = p_kgms[0] / MC_KGMS
    uy = p_kgms[1] / MC_KGMS
    uz = p_kgms[2] / MC_KGMS

    eps_scale = (_Q_ELECTRON * dt_s) / (2.0 * MC_KGMS)
    tau_scale = (_Q_ELECTRON * dt_s) / (2.0 * M_ELECTRON)

    ex_eps = eps_scale * E_vm[0]
    ey_eps = eps_scale * E_vm[1]
    ez_eps = eps_scale * E_vm[2]

    umx = ux + ex_eps
    umy = uy + ey_eps
    umz = uz + ez_eps

    gamma_minus = math.sqrt(1.0 + umx * umx + umy * umy + umz * umz)
    inv_gm = tau_scale / gamma_minus

    tx = B_t[0] * inv_gm
    ty = B_t[1] * inv_gm
    tz = B_t[2] * inv_gm

    s_fac = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
    sx = s_fac * tx
    sy = s_fac * ty
    sz = s_fac * tz

    upx = umx + (umy * tz - umz * ty)
    upy = umy + (umz * tx - umx * tz)
    upz = umz + (umx * ty - umy * tx)

    uplx = umx + (upy * sz - upz * sy)
    uply = umy + (upz * sx - upx * sz)
    uplz = umz + (upx * sy - upy * sx)

    unx = uplx + ex_eps
    uny = uply + ey_eps
    unz = uplz + ez_eps

    gamma_next = math.sqrt(1.0 + unx * unx + uny * uny + unz * unz)

    p_next = np.empty(3, dtype=np.float64)
    p_next[0] = MC_KGMS * unx
    p_next[1] = MC_KGMS * uny
    p_next[2] = MC_KGMS * unz

    v_dt = (C_LIGHT * dt_s) / gamma_next
    r_next = np.empty(3, dtype=np.float64)
    r_next[0] = r_m[0] + unx * v_dt
    r_next[1] = r_m[1] + uny * v_dt
    r_next[2] = r_m[2] + unz * v_dt

    return r_next, p_next


@njit(cache=True)
def _propagate_ensemble_numba(
    r0_m: np.ndarray,
    p0_kgms: np.ndarray,
    dt_jitter_s: np.ndarray,
    t_s: np.ndarray,
    dt_s: float,
    use_vay: bool,
    signed_alpha: bool,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_particles = r0_m.shape[0]
    n_steps = t_s.shape[0]

    r_trace = np.empty((n_particles, n_steps, 3), dtype=np.float64)
    p_trace = np.empty((n_particles, n_steps, 3), dtype=np.float64)
    gamma_trace = np.empty((n_particles, n_steps), dtype=np.float64)
    chi_trace = np.empty((n_particles, n_steps), dtype=np.float64)
    alpha_trace = np.empty((n_particles, n_steps), dtype=np.float64)

    for i in range(n_particles):
        r = r0_m[i].copy()
        p = p0_kgms[i].copy()
        jitter = dt_jitter_s[i]

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

            r_trace[i, n, 0] = r[0]
            r_trace[i, n, 1] = r[1]
            r_trace[i, n, 2] = r[2]
            p_trace[i, n, 0] = p[0]
            p_trace[i, n, 1] = p[1]
            p_trace[i, n, 2] = p[2]
            gamma_trace[i, n] = gamma_curr
            chi_trace[i, n] = compute_chi_numba(e_vm, b_t, p)
            alpha_trace[i, n] = compute_deviation_angle_numba(p, signed_z=signed_alpha)

            if n < n_steps - 1:
                if use_vay:
                    r, p = step_vay_numba(r, p, e_vm, b_t, dt_s)
                else:
                    r, p = step_boris_numba(r, p, e_vm, b_t, dt_s)

    return r_trace, p_trace, gamma_trace, chi_trace, alpha_trace


def propagate_ensemble(
    pack: FieldPack,
    batch: ElectronBatch,
    dt_s: float | None = None,
    t_start_s: float | None = None,
    t_end_s: float | None = None,
    method: str = "vay",
    signed_alpha: bool = False,
) -> TrajectoryResult:
    """Propagate an ElectronBatch through a FieldPack without QED recoil."""
    if method not in ("vay", "boris"):
        raise ValueError(f"method must be 'vay' or 'boris', got {method!r}")
    if batch.r0_m.shape[0] < 1:
        raise ValueError("ElectronBatch must contain at least 1 electron")

    t0_s = period_from_lambda(pack.lambda0_m)
    dt_max_s = t0_s / 20.0
    dt_eff = (t0_s / 40.0) if dt_s is None else float(dt_s)
    if not math.isfinite(dt_eff) or dt_eff <= 0.0 or dt_eff > dt_max_s:
        raise ValueError(
            f"dt_s ({dt_eff:.6e} s) must satisfy 0 < dt_s <= T0/20 ({dt_max_s:.6e} s)"
        )

    v0_y = C_LIGHT * math.sqrt(1.0 - 1.0 / (batch.gamma0 * batch.gamma0))
    y_start = float(batch.r0_m[0, 1])
    y_grid_end = pack.grid.y0_m + (pack.grid.Ny - 1) * pack.grid.dy_m

    t_start = (pack.pulse.t_ref_s + y_start / v0_y) if t_start_s is None else float(t_start_s)
    if t_end_s is None:
        t_end = pack.pulse.t_ref_s + y_grid_end / v0_y
        if t_end <= t_start:
            t_end = t_start + ((pack.grid.Ny - 1) * pack.grid.dy_m) / v0_y
    else:
        t_end = float(t_end_s)

    if not (math.isfinite(t_start) and math.isfinite(t_end)) or t_end <= t_start:
        raise ValueError(f"t_end_s ({t_end}) must be finite and greater than t_start_s ({t_start})")

    n_steps = max(2, int(math.ceil((t_end - t_start) / dt_eff)) + 1)
    t_s = t_start + np.arange(n_steps, dtype=np.float64) * dt_eff

    use_gauss_env = (pack.freqs.size == 1) and (pack.pulse.model == "gauss")
    sigma_t = pack.pulse.sigma_t_s if use_gauss_env else 1.0

    r_trace, p_trace, gamma_trace, chi_trace, alpha_trace = _propagate_ensemble_numba(
        np.ascontiguousarray(batch.r0_m, dtype=np.float64),
        np.ascontiguousarray(batch.p0_kgms, dtype=np.float64),
        np.ascontiguousarray(batch.dt_jitter_s, dtype=np.float64),
        t_s,
        dt_eff,
        method == "vay",
        bool(signed_alpha),
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
    )

    return TrajectoryResult(
        t_s=t_s,
        r_m=r_trace,
        p_kgms=p_trace,
        gamma=gamma_trace,
        chi=chi_trace,
        alpha=alpha_trace,
        batch=batch,
    )
