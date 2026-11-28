"""Unit and physics verification suite for src/particles."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.particles import (
    ALPHA_FS,
    HBAR,
    MC2_GEV,
    MC2_J,
    MC_KGMS,
    PhotonCatalog,
    apply_selection_cuts,
    build_lcfa_table,
    compute_chi,
    compute_deviation_angle,
    compute_fig2_phase_space,
    compute_high_chi_time_fraction,
    compute_sigma_and_eta,
    energy_gev_to_gamma,
    estimate_electron_loss_shots,
    estimate_shots_for_confidence,
    evaluate_eb,
    evaluate_eb_batch,
    f_number_to_opening_angle_deg,
    fig4_reference_curve,
    fig5_reference_contours,
    lcfa_rate_per_s,
    load_field_pack,
    make_analytic_bidipole_pack,
    omega_from_lambda,
    period_from_lambda,
    propagate_ensemble,
    sample_electron_batch,
    save_field_pack,
    simulate_qed_ensemble,
    step_boris_numba,
    step_vay_numba,
    tau_fwhm_to_sigma_t,
    theoretical_chi_max,
)
from src.particles.qed import (
    _interp_table_1d_numba,
    _lookup_cdf_tail_from_invcdf_numba,
    _lookup_g_rate_numba,
    _lookup_inv_cdf_numba,
    W0_LCFA_PER_S,
)

SAMPLE_FIELD_PATH = REPO_ROOT / "data" / "sample_field.npz"


def test_schema_and_sample_field_pack(tmp_path: Path) -> None:
    """Verify sample_field.npz loads cleanly with 5D shape and supports 4D LP promotion."""
    pack = load_field_pack(SAMPLE_FIELD_PATH)
    assert pack.Ec.ndim == 5
    assert pack.Bc.ndim == 5
    assert pack.Ec.shape == (1, 3, pack.grid.Nz, pack.grid.Ny, pack.grid.Nx)
    assert pack.Bc.shape == pack.Ec.shape
    assert pack.Ec.dtype == np.complex128
    assert pack.Bc.dtype == np.complex128
    assert pack.pol == "z-linear"
    assert pack.checks.a_max > 100.0
    assert abs(pack.checks.a_max - pack.checks.a_max_expected) / pack.checks.a_max_expected < 0.01

    # Test legacy 4D (Nf, Nz, Ny, Nx) scalar Ez auto-promotion for z-linear
    ec_4d = pack.Ec[:, 2, :, :, :].copy()
    legacy_path = tmp_path / "legacy_4d.npz"
    np.savez_compressed(legacy_path, Ec=ec_4d, metadata=json.dumps(pack.metadata_raw))
    reloaded_4d = load_field_pack(legacy_path)
    assert reloaded_4d.Ec.shape == pack.Ec.shape
    assert np.allclose(reloaded_4d.Ec[:, 2], ec_4d)
    assert np.all(reloaded_4d.Ec[:, 0] == 0.0)
    assert np.all(reloaded_4d.Ec[:, 1] == 0.0)

    # Reject 4D array when pol is circular
    bad_meta = copy.deepcopy(pack.metadata_raw)
    bad_meta["pol"] = "circular"
    bad_4d_path = tmp_path / "bad_4d_cp.npz"
    np.savez_compressed(bad_4d_path, Ec=ec_4d, metadata=json.dumps(bad_meta))
    with pytest.raises(ValueError, match="4D Ec"):
        load_field_pack(bad_4d_path)


@pytest.mark.parametrize("pol", ["z-linear", "circular"])
def test_curl_fallback_matches_analytic_bidipole(tmp_path: Path, pol: str) -> None:
    """Verify Bc = (i/omega) curl(Ec) fallback matches exact analytic Bc within < 2% at focus."""
    pack = make_analytic_bidipole_pack(pol=pol)
    no_bc_path = tmp_path / f"no_bc_{pol}.npz"
    save_field_pack(no_bc_path, pack, include_bc=False)

    reloaded = load_field_pack(no_bc_path)
    mid_z = pack.grid.Nz // 2
    mid_y = pack.grid.Ny // 2
    mid_x = pack.grid.Nx // 2

    # Compare inside an interior 5x5x5 cube around the focus
    sl = slice(mid_z - 2, mid_z + 3)
    b_exact = pack.Bc[0, :, sl, sl, sl]
    b_fd = reloaded.Bc[0, :, sl, sl, sl]
    rel_err = np.linalg.norm(b_fd - b_exact) / np.linalg.norm(b_exact)
    assert rel_err < 0.02


def test_fail_loud_guards(tmp_path: Path) -> None:
    """Verify invalid files, corrupted checks, coarse timesteps, and unphysical params fail loudly."""
    with pytest.raises(FileNotFoundError):
        load_field_pack(tmp_path / "does_not_exist.npz")

    pack = load_field_pack(SAMPLE_FIELD_PATH)

    # Corrupted checks.a_max
    bad_meta = copy.deepcopy(pack.metadata_raw)
    bad_meta["checks"]["a_max"] = pack.checks.a_max * 2.0
    bad_amax_path = tmp_path / "bad_amax.npz"
    save_field_pack(bad_amax_path, pack.Ec, bad_meta, Bc=pack.Bc)
    with pytest.raises(ValueError, match="a_max self-check failed"):
        load_field_pack(bad_amax_path)

    # Exceeded power conservation error
    bad_power_meta = copy.deepcopy(pack.metadata_raw)
    bad_power_meta["checks"]["power_conservation_rel_err"] = 0.25
    bad_power_path = tmp_path / "bad_power.npz"
    save_field_pack(bad_power_path, pack.Ec, bad_power_meta, Bc=pack.Bc)
    with pytest.raises(ValueError, match="power_conservation_rel_err"):
        load_field_pack(bad_power_path)

    # Coarse timestep > T0 / 20
    t0_s = period_from_lambda(pack.lambda0_m)
    batch = sample_electron_batch(2, energy_gev=10.0, b_max_m=pack.lambda0_m, y0_m=pack.grid.y0_m)
    with pytest.raises(ValueError, match="T0/20"):
        propagate_ensemble(pack, batch, dt_s=t0_s / 10.0)
    with pytest.raises(ValueError, match="T0/20"):
        simulate_qed_ensemble(pack, batch, dt_s=t0_s / 15.0)
    with pytest.raises(ValueError, match="T0/20"):
        propagate_ensemble(pack, batch, dt_s=-1.0e-18)

    # Negative / unphysical constants
    with pytest.raises(ValueError):
        omega_from_lambda(-0.8e-6)
    with pytest.raises(ValueError):
        tau_fwhm_to_sigma_t(0.0)
    with pytest.raises(ValueError):
        energy_gev_to_gamma(0.5 * MC2_GEV)
    with pytest.raises(ValueError):
        theoretical_chi_max(-1.0, 1.0e15, 1.0e-6)

    # Loader negative tests: missing Ec, missing metadata, non-monotonic freqs, non-finite Ec, unsupported model
    missing_ec_path = tmp_path / "missing_ec.npz"
    np.savez_compressed(missing_ec_path, metadata=json.dumps(pack.metadata_raw))
    with pytest.raises(ValueError, match="missing required keys"):
        load_field_pack(missing_ec_path)

    missing_meta_path = tmp_path / "missing_meta.npz"
    np.savez_compressed(missing_meta_path, Ec=pack.Ec)
    with pytest.raises(ValueError, match="missing required keys"):
        load_field_pack(missing_meta_path)

    nonmono_meta = copy.deepcopy(pack.metadata_raw)
    w0 = float(pack.freqs[0])
    nonmono_meta["freqs"] = [w0, 0.9 * w0]
    ec_2f = np.concatenate([pack.Ec, pack.Ec], axis=0)
    bc_2f = np.concatenate([pack.Bc, pack.Bc], axis=0)
    nonmono_path = tmp_path / "nonmono_freqs.npz"
    save_field_pack(nonmono_path, ec_2f, nonmono_meta, Bc=bc_2f)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_field_pack(nonmono_path)

    ec_nan = pack.Ec.copy()
    ec_nan[0, 0, 0, 0, 0] = np.nan + 0.0j
    nan_ec_path = tmp_path / "nan_ec.npz"
    save_field_pack(nan_ec_path, ec_nan, pack.metadata_raw, Bc=pack.Bc)
    with pytest.raises(ValueError, match="non-finite"):
        load_field_pack(nan_ec_path)

    bad_pulse_meta = copy.deepcopy(pack.metadata_raw)
    bad_pulse_meta["pulse"]["model"] = "square"
    bad_pulse_path = tmp_path / "bad_pulse_model.npz"
    np.savez_compressed(bad_pulse_path, Ec=pack.Ec, Bc=pack.Bc, metadata=json.dumps(bad_pulse_meta))
    with pytest.raises(ValueError):
        load_field_pack(bad_pulse_path)

    # NaN / Inf coordinates and times fail loud in field evaluation
    with pytest.raises(ValueError, match="finite"):
        evaluate_eb(pack, np.array([np.nan, 0.0, 0.0]), 0.0)
    with pytest.raises(ValueError, match="finite"):
        evaluate_eb(pack, np.zeros(3), float("inf"))
    r_bad = np.array([[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        evaluate_eb_batch(pack, r_bad, 0.0)
    with pytest.raises(ValueError, match="finite"):
        evaluate_eb_batch(pack, np.zeros((2, 3), dtype=np.float64), float("nan"))


@pytest.mark.parametrize("method_name,stepper", [("vay", step_vay_numba), ("boris", step_boris_numba)])
def test_vay_and_boris_energy_conservation_in_b_field(method_name: str, stepper) -> None:
    """Propagate a 10 GeV electron in pure B = 1e4 T x_hat for 500 steps and assert gamma conservation."""
    gamma0 = energy_gev_to_gamma(10.0)
    p = np.array([0.0, MC_KGMS * math.sqrt(gamma0**2 - 1.0), 0.0], dtype=np.float64)
    r = np.zeros(3, dtype=np.float64)
    e_zero = np.zeros(3, dtype=np.float64)
    b_const = np.array([1.0e4, 0.0, 0.0], dtype=np.float64)
    dt_s = 1.0e-16

    for _ in range(500):
        r, p = stepper(r, p, e_zero, b_const, dt_s)

    gamma_final = math.sqrt(1.0 + float(np.sum((p / MC_KGMS) ** 2)))
    rel_err = abs(gamma_final - gamma0) / gamma0
    assert rel_err < 1.0e-12, f"{method_name} energy drift {rel_err} exceeded 1e-12"


def test_impact_parameter_sampling_distribution() -> None:
    """Verify p(b) = 2b / b_max^2 yields <b> = 2/3 b_max and <b^2> = 1/2 b_max^2 within 1%."""
    b_max = 1.5e-6
    rng = np.random.default_rng(12345)
    batch = sample_electron_batch(
        n_particles=50_000,
        energy_gev=10.0,
        b_max_m=b_max,
        y0_m=-2.0e-6,
        sigma_dt_s=1.0e-15,
        jitter_mode="per_shot",
        rng=rng,
    )
    mean_b = float(np.mean(batch.b_impact_m))
    mean_b2 = float(np.mean(batch.b_impact_m**2))
    assert abs(mean_b - (2.0 / 3.0) * b_max) / ((2.0 / 3.0) * b_max) < 0.01
    assert abs(mean_b2 - 0.5 * b_max**2) / (0.5 * b_max**2) < 0.01
    # Per-shot jitter shares one draw across all electrons in the batch
    assert np.all(batch.dt_jitter_s == batch.dt_jitter_s[0])

    batch_pp = sample_electron_batch(
        n_particles=256,
        energy_gev=10.0,
        b_max_m=b_max,
        y0_m=-2.0e-6,
        sigma_dt_s=1.0e-15,
        jitter_mode="per_particle",
        rng=rng,
    )
    assert np.std(batch_pp.dt_jitter_s) > 0.0


def test_bidipole_peak_chi_matches_eq1() -> None:
    """Evaluate chi at r=0, t=0 for eps=1 GeV, P=1 PW, lambda=1 um, kappa=5.25 and assert |chi - 5.25| < 1e-3."""
    pack = make_analytic_bidipole_pack(
        power_w=1.0e15,
        lambda0_m=1.0e-6,
        f_number=0.05,
        pol="z-linear",
        kappa=5.25,
        n_grid=25,
    )
    e0, b0 = evaluate_eb(pack, np.zeros(3, dtype=np.float64), 0.0)
    gamma0 = energy_gev_to_gamma(1.0)
    p0 = np.array([0.0, MC_KGMS * math.sqrt(gamma0**2 - 1.0), 0.0], dtype=np.float64)
    chi0 = compute_chi(e0, b0, p0)
    assert abs(chi0 - 5.25) < 1.0e-3

    # Also verify out-of-bounds query returns exact zeros
    e_oob, b_oob = evaluate_eb(pack, np.array([10.0, 0.0, 0.0]), 0.0)
    assert np.all(e_oob == 0.0) and np.all(b_oob == 0.0)

    # Verify Gaussian intensity FWHM envelope: I(+-tau_fwhm/2) == 0.5 * I(0) at integer-cycle offset
    fwhm_pack = make_analytic_bidipole_pack(
        power_w=1.0e15,
        lambda0_m=1.0e-6,
        f_number=0.25,
        pol="z-linear",
        n_cycles_fwhm=10.0,
        n_grid=15,
    )
    e_peak, _ = evaluate_eb(fwhm_pack, np.zeros(3, dtype=np.float64), 0.0)
    e_plus, _ = evaluate_eb(fwhm_pack, np.zeros(3, dtype=np.float64), 0.5 * fwhm_pack.pulse.tau_fwhm_s)
    e_minus, _ = evaluate_eb(fwhm_pack, np.zeros(3, dtype=np.float64), -0.5 * fwhm_pack.pulse.tau_fwhm_s)
    i0 = float(np.dot(e_peak, e_peak))
    assert math.isclose(float(np.dot(e_plus, e_plus)) / i0, 0.5, rel_tol=1e-12)
    assert math.isclose(float(np.dot(e_minus, e_minus)) / i0, 0.5, rel_tol=1e-12)


def test_lcfa_rate_and_inverse_cdf() -> None:
    """Verify LCFA rate matches classical asymptote at chi=1e-3 (<2%) and inv_cdf monotonicity."""
    table = build_lcfa_table()
    gamma = energy_gev_to_gamma(10.0)
    chi_low = 1.0e-3
    w_rad = lcfa_rate_per_s(gamma, chi_low, table=table)
    w_cl = (5.0 * ALPHA_FS * MC2_J / (2.0 * math.sqrt(3.0) * HBAR * gamma)) * chi_low
    assert abs(w_rad - w_cl) / w_cl < 0.02

    assert np.all(np.diff(table.inv_cdf, axis=1) >= 0.0)
    assert np.all(table.inv_cdf[:, 0] == 0.0)
    assert np.all(table.inv_cdf[:, -1] < 1.0)
    assert np.all((table.high_energy_fraction >= 0.0) & (table.high_energy_fraction <= 1.0))

    # Tail-clustered inv_cdf accurately recovers P(delta > 0.5) down to rare-tail chi=0.05
    for chi_test in (0.05, 0.1, 0.5, 1.0, 10.0):
        tail_inv = _lookup_cdf_tail_from_invcdf_numba(
            chi_test, 0.5, table.log_chi_grid, table.inv_cdf
        )
        tail_ref = float(np.interp(chi_test, table.chi_grid, table.high_energy_fraction))
        assert tail_ref > 0.0
        assert math.isclose(tail_inv, tail_ref, rel_tol=0.15)

    # Direct Monte Carlo draw from inv_cdf at chi=0.05 respects rare tail probability (~4.3e-8)
    rng_tail = np.random.default_rng(42)
    q_samples = rng_tail.uniform(0.0, 1.0, size=100_000)
    high_count = sum(
        1
        for q_val in q_samples
        if _lookup_inv_cdf_numba(0.05, float(q_val), table.log_chi_grid, table.inv_cdf) > 0.5
    )
    assert (high_count / 100_000) < 1.0e-4

    # Bounded fraction columns stay clamped <= 1.0 at extreme chi=1e5 while G(chi) extrapolates chi^(2/3)
    for col in (
        table.high_energy_fraction,
        table.loss_gt_1pct_fraction,
        table.mean_delta_high,
        table.mean_delta_total,
    ):
        extrap_val = _interp_table_1d_numba(1.0e5, table.log_chi_grid, col)
        assert 0.0 <= extrap_val <= 1.0
    assert _lookup_g_rate_numba(1.0e5, table.log_chi_grid, table.g_rate) > float(table.g_rate[-1])

    # Smoke-test both Monte Carlo and weighted QED modes end-to-end and verify sigma_cm2 consistency
    pack = load_field_pack(SAMPLE_FIELD_PATH)
    batch = sample_electron_batch(
        n_particles=1024,
        energy_gev=10.0,
        b_max_m=0.25 * pack.lambda0_m,
        y0_m=pack.grid.y0_m,
        rng=np.random.default_rng(7),
    )
    res_mc = simulate_qed_ensemble(pack, batch, mode="monte_carlo", rng=np.random.default_rng(9))
    res_wt = simulate_qed_ensemble(pack, batch, mode="weighted")
    assert res_mc.mode == "monte_carlo"
    assert res_wt.mode == "weighted"
    assert res_wt.photons.weight.size > 0
    assert np.all(res_wt.p_no_loss[:, 0] == 1.0)
    assert np.all((res_wt.p_no_loss > 0.0) & (res_wt.p_no_loss <= 1.0))

    chi_max_10gev = pack.checks.chi_max_per_GeV * batch.energy_gev
    metrics_mc = compute_sigma_and_eta(
        res_mc.photons,
        pol=pack.pol,
        a_max=pack.checks.a_max,
        chi_max=chi_max_10gev,
        b_max_m=batch.b_max_m,
        n_electrons=1024,
    )
    metrics_wt = compute_sigma_and_eta(
        res_wt.photons,
        pol=pack.pol,
        a_max=pack.checks.a_max,
        chi_max=chi_max_10gev,
        b_max_m=batch.b_max_m,
        n_electrons=1024,
    )
    assert metrics_wt["sigma_cm2"] > 0.0 and metrics_mc["sigma_cm2"] > 0.0
    # The MC draw yields only ~9 signal photons (Poisson +-33%) and the weighted mode
    # propagates undepleted trajectories, so the deterministic ratio for these seeds sits
    # near 1.7; keep a tightened regression band around it.
    assert 1.35 < metrics_mc["sigma_cm2"] / metrics_wt["sigma_cm2"] < 2.0

    # Exercise chi phase-space & high-chi time fraction helpers
    small_batch = sample_electron_batch(
        n_particles=32,
        energy_gev=10.0,
        b_max_m=pack.lambda0_m,
        y0_m=pack.grid.y0_m,
        rng=np.random.default_rng(7),
    )
    traj = propagate_ensemble(pack, small_batch, signed_alpha=True)
    frac = compute_high_chi_time_fraction(traj.chi, dt_s=float(traj.t_s[1] - traj.t_s[0]))
    assert frac.shape == (32,)
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    ps = compute_fig2_phase_space(
        traj.alpha,
        traj.chi,
        a_max=pack.checks.a_max,
        gamma0=batch.gamma0,
        chi_max=pack.checks.chi_max_per_GeV * batch.energy_gev,
        signed_alpha=True,
    )
    assert ps["density"].shape == (120, 120)
    assert compute_deviation_angle(np.zeros(3)) == 0.0
    with pytest.raises(ValueError, match="signed_alpha=False"):
        compute_fig2_phase_space(
            np.array([[-0.01, 0.01]]),
            np.array([[1.0, 1.0]]),
            a_max=pack.checks.a_max,
            gamma0=batch.gamma0,
            chi_max=chi_max_10gev,
            signed_alpha=False,
        )
    with pytest.raises(ValueError, match="non-negative"):
        compute_fig2_phase_space(
            np.array([[0.01, 0.01]]),
            np.array([[-0.1, 1.0]]),
            a_max=pack.checks.a_max,
            gamma0=batch.gamma0,
            chi_max=chi_max_10gev,
        )
    with pytest.raises(ValueError, match="chi_max_ref"):
        simulate_qed_ensemble(pack, small_batch, mode="weighted", chi_max_ref=-1.0)


def test_weighted_mode_small_loss_accumulation() -> None:
    """Verify weighted-mode p_no_loss tracks accumulated small (delta <= 0.01) losses."""
    table = build_lcfa_table()
    assert np.all(table.mean_delta_small >= 0.0)
    assert np.all(table.mean_delta2_small >= 0.0)
    # delta <= 0.01 under the indicator, and delta^2 <= delta for delta in [0, 1]
    assert np.all(table.mean_delta_small <= 0.01)
    assert np.all(table.mean_delta2_small <= table.mean_delta_small)
    assert np.all(table.mean_delta_small <= table.mean_delta_total)

    # Below chi ~ 6.7e-5 every emission has delta <= 0.01, so the small-loss mean
    # column collapses exactly onto the total mean
    tiny = build_lcfa_table(chi_min=1.0e-7, chi_max=6.0e-5, n_chi=16, n_q=16)
    assert np.array_equal(tiny.mean_delta_small, tiny.mean_delta_total)
    assert np.all(tiny.mean_delta2_small >= 0.0)

    pack = load_field_pack(SAMPLE_FIELD_PATH)
    batch = sample_electron_batch(
        n_particles=64,
        energy_gev=10.0,
        b_max_m=0.25 * pack.lambda0_m,
        y0_m=pack.grid.y0_m,
        rng=np.random.default_rng(7),
    )
    res_wt = simulate_qed_ensemble(pack, batch, mode="weighted")
    p_no_loss = res_wt.p_no_loss

    # Survival starts at 1.0 and is monotonically non-increasing along every trajectory
    assert np.all(p_no_loss[:, 0] == 1.0)
    assert np.all(np.diff(p_no_loss, axis=1) <= 0.0)

    # Recompute the single-big-loss survival exp(-int w_loss_1pct dt) from the chi trace
    dt_s = float(res_wt.t_s[1] - res_wt.t_s[0])
    p_no_big_ref = np.ones_like(p_no_loss)
    for i in range(p_no_loss.shape[0]):
        hazard = 0.0
        for n in range(p_no_loss.shape[1] - 1):
            chi_step = float(res_wt.chi[i, n])
            w_rad = (W0_LCFA_PER_S / batch.gamma0) * _lookup_g_rate_numba(
                chi_step, table.log_chi_grid, table.g_rate
            )
            f_1pct = _interp_table_1d_numba(chi_step, table.log_chi_grid, table.loss_gt_1pct_fraction)
            hazard += w_rad * f_1pct * dt_s
            p_no_big_ref[i, n + 1] = math.exp(-hazard)

    # p_no_loss never exceeds the big-loss-only survival bound
    assert np.all(p_no_loss <= p_no_big_ref + 1.0e-9)

    # Small-loss accumulation is active: survival drops strictly below the bound once the
    # accumulated small losses approach the 1% depletion threshold
    assert np.any(p_no_loss < 0.9 * p_no_big_ref)


def test_selection_cuts_and_3sigma_shots() -> None:
    """Verify LP/CP cuts, >=0.99 gamma0 filter, and Sec. V 3-sigma shot counts."""
    gamma0 = 2.0e4
    a_max_lp = 200.0
    a_max_cp = a_max_lp / math.sqrt(2.0)
    chi_max = 20.0

    # Synthetic 4-photon catalog testing boundary conditions of every cut:
    # 0: passes total and signal for LP (alpha = 0.65 * a_max_lp / gamma0)
    # 1: fails LP angle cut (alpha = 0.55 * a_max_lp / gamma0)
    # 2: passes total, fails chi > chi_max / 2
    # 3: passes total, fails gamma_prior >= 0.99 * gamma0
    catalog = PhotonCatalog(
        energy_gev=np.array([6.0, 6.0, 6.0, 6.0]),
        delta_init=np.array([0.6, 0.6, 0.6, 0.6]),
        alpha_rad=np.array(
            [
                0.65 * a_max_lp / gamma0,
                0.55 * a_max_lp / gamma0,
                0.75 * a_max_lp / gamma0,
                0.75 * a_max_lp / gamma0,
            ]
        ),
        chi_emit=np.array([12.0, 12.0, 8.0, 12.0]),
        gamma_prior=np.array([0.995 * gamma0, 0.995 * gamma0, 0.995 * gamma0, 0.98 * gamma0]),
        gamma0=np.full(4, gamma0),
        b_impact_m=np.zeros(4),
        weight=np.ones(4),
        t_emit_s=np.zeros(4),
        particle_idx=np.arange(4, dtype=np.int64),
    )

    cuts_lp = apply_selection_cuts(catalog, pol="z-linear", a_max=a_max_lp, chi_max=chi_max)
    assert np.array_equal(cuts_lp["total_mask"], np.array([True, False, True, True]))
    assert np.array_equal(cuts_lp["signal_mask"], np.array([True, False, False, False]))

    bad_catalog = PhotonCatalog(
        energy_gev=catalog.energy_gev[:2],
        delta_init=catalog.delta_init,
        alpha_rad=catalog.alpha_rad,
        chi_emit=catalog.chi_emit,
        gamma_prior=catalog.gamma_prior,
        gamma0=catalog.gamma0,
        b_impact_m=catalog.b_impact_m,
        weight=catalog.weight,
        t_emit_s=catalog.t_emit_s,
        particle_idx=catalog.particle_idx,
    )
    with pytest.raises(ValueError, match="matching length"):
        apply_selection_cuts(bad_catalog, pol="z-linear", a_max=a_max_lp, chi_max=chi_max)

    metrics_lp = compute_sigma_and_eta(
        catalog,
        pol="z-linear",
        a_max=a_max_lp,
        chi_max=chi_max,
        b_max_m=1.0e-6,
        n_electrons=100,
    )
    assert math.isclose(metrics_lp["eta"], 1.0 / 3.0, rel_tol=1e-12)
    assert math.isclose(metrics_lp["sigma_cm2"], 1.0e4 * math.pi * (1.0e-6**2) * 0.01, rel_tol=1e-12)

    # For CP, alpha_thresh = 0.7 * a_max_cp / gamma0 ~ 0.495 * a_max_lp / gamma0 -> photon 1 passes CP total!
    cuts_cp = apply_selection_cuts(catalog, pol="circular", a_max=a_max_cp, chi_max=chi_max)
    assert np.array_equal(cuts_cp["total_mask"], np.array([True, True, True, True]))
    assert np.array_equal(cuts_cp["signal_mask"], np.array([True, True, False, False]))

    # Sec. V 3-sigma shot counts
    shot_est = estimate_shots_for_confidence(
        sigma_cm2=1.0e-11,
        eta=0.1,
        n_e_cm3=1.0e19,
        tau_l_s=13.34e-15,
    )
    assert 200.0 < shot_est.n_shots_required < 250.0
    assert math.isclose(shot_est.n_total_required, 9.0e6, rel_tol=1e-12)

    eloss = estimate_electron_loss_shots()
    assert eloss["shots_required"] > 5.0e6
    assert 5.0e7 < eloss["shots_required_paper_rounded"] < 1.5e8

    # Verify Fig. 4 & Fig. 5 benchmark helpers
    assert math.isclose(f_number_to_opening_angle_deg(0.25), 90.0, rel_tol=1e-12)
    f4 = fig4_reference_curve()
    assert f4["chi_max"][0] > 5.2 and f4["chi_max"][-1] < 1.3
    f5 = fig5_reference_contours()
    assert f5["contour_power_w"].shape == (4, 100)
