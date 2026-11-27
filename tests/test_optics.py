"""Tests for the optics module (bi-dipole focusing simulation).

Run with ``pytest -q`` (the project's test task) or directly with
``python tests/test_optics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optics import cli  # noqa: E402
from src.optics import comparison  # noqa: E402
from src.optics import constants as C  # noqa: E402
from src.optics import focus  # noqa: E402
from src.optics import metadata as meta_mod  # noqa: E402
from src.optics import sweep  # noqa: E402

LAM = 0.8e-6
F_NUMBER = 0.25
POWER = 1.0e15


def _compute_small():
    Ec, Bc, grid = focus.focal_field_angular_spectrum(
        lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER,
        nx=32, ny=32, nz=32, span_xy_m=6.4e-6, span_y_m=3.2e-6,
        ntheta=40, nphi=80,
    )
    checks = focus.compute_checks(Ec, lambda0_m=LAM, power_w=POWER, grid=grid, Bc=Bc)
    return Ec, Bc, grid, checks


def _sample_n(ntheta=40, nphi=12):
    th = np.linspace(0.05, np.pi / 2 - 0.05, ntheta)
    ph = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    return np.stack([np.sin(TH) * np.cos(PH), -np.cos(TH), np.sin(TH) * np.sin(PH)], axis=-1)


def test_metadata_validates_against_schema():
    Ec, Bc, grid, checks = _compute_small()
    meta = meta_mod.build_metadata(
        lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER, pol="z-linear",
        grid=grid, checks=checks, tau_fwhm_s=30.0e-15,
    )
    meta_mod.validate(meta)


def test_power_conservation():
    Ec, Bc, grid, checks = _compute_small()
    p0 = focus.field_power_through_y_plane(Ec[0], Bc[0], grid["dx_m"], grid["dz_m"])
    assert p0 < 0.0
    assert 0.0 < checks["power_conservation_rel_err"] < 0.10


def test_curl_b_matches_direct_b():
    # Cross-check: (i/omega) curl E must match the direct Debye B within grid error.
    Ec, Bc, grid = focus.focal_field_angular_spectrum(
        lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER,
        nx=32, ny=32, nz=32, span_xy_m=3.2e-6, span_y_m=3.2e-6,
        ntheta=40, nphi=80,
    )
    omega = C.omega_from_lambda(LAM)
    Bc_curl = focus.magnetic_field_from_curl(
        Ec[0], grid["dx_m"], grid["dy_m"], grid["dz_m"], omega
    )
    rel = float(np.linalg.norm(Bc_curl - Bc[0]) / np.linalg.norm(Bc[0]))
    assert rel < 0.15


def test_global_phase_aligned_at_focus():
    nx = ny = nz = 16
    for far_field in (
        focus.bi_dipole_far_field,
        lambda n: focus.gaussian_far_field(n, f_number=F_NUMBER),
    ):
        Ec, Bc, _ = focus.focal_field_angular_spectrum(
            lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER,
            nx=nx, ny=ny, nz=nz, span_xy_m=3.2e-6, span_y_m=3.2e-6,
            ntheta=24, nphi=48, far_field=far_field,
        )
        ec_focus = Ec[0, 2, nz // 2, ny // 2, nx // 2]
        bc_focus = Bc[0, 0, nz // 2, ny // 2, nx // 2]
        assert ec_focus.real > 0.0
        assert abs(ec_focus.imag) < 1e-9 * abs(ec_focus.real)
        assert bc_focus.real < 0.0
        assert abs(bc_focus.imag) < 1e-9 * abs(bc_focus.real)


def test_a_max_close_to_expected():
    _, _, _, checks = _compute_small()
    assert 0.85 * checks["a_max_expected"] < checks["a_max"] < 1.15 * checks["a_max_expected"]


def test_chi_max_matches_a_max_scaling():
    _, _, _, checks = _compute_small()
    omega = C.omega_from_lambda(LAM)
    hbar_omega_over_mc2 = C.HBAR * omega / C.MC2_J
    expected = 2.0 * checks["a_max"] * C.GAMMA_PER_GEV * hbar_omega_over_mc2
    assert abs(checks["chi_max_per_GeV"] - expected) / expected < 0.05


def test_peak_location_at_focus():
    _, _, _, checks = _compute_small()
    x, y, z = checks["max_location_m"]
    assert abs(x) < 1e-7 and abs(y) < 1e-7 and abs(z) < 1e-7


def test_kappa_sweep_saturates_and_decreases():
    fns = [0.25, 0.5, 1.0, 2.0]
    res = sweep.kappa_eff_vs_f_number(
        fns, lambda0_m=1.0e-6, power_w=POWER, energy_gev=1.0, ntheta=128, nphi=256,
    )
    chi = [r["chi_max_per_GeV"] for r in res]
    assert abs(chi[0] - 5.25) / 5.25 < 0.15
    assert chi[0] > chi[1] > chi[2] > chi[3]


def test_far_field_is_transverse():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(200, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    E = focus.bi_dipole_far_field(n)
    assert np.max(np.abs(np.sum(n * E, axis=-1))) < 1e-12


def test_ideal_reflection_matches_analytic():
    # Reflecting the ideal I^p(R) profile must reproduce the analytic far field.
    n = _sample_n()
    a = focus.bi_dipole_far_field(n).reshape(-1, 3)
    b = focus.ideal_far_field(n, f_number=F_NUMBER).reshape(-1, 3)
    a = a / np.sqrt(np.sum(np.abs(a) ** 2))
    b = b / np.sqrt(np.sum(np.abs(b) ** 2))
    overlap = abs(np.vdot(a.ravel(), b.ravel()))
    assert overlap > 0.999


def test_ideal_reflection_handles_f_below_quarter():
    th = np.linspace(0.05, 2.2, 40)
    ph = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(TH) * np.cos(PH), -np.cos(TH), np.sin(TH) * np.sin(PH)], axis=-1)
    a = focus.bi_dipole_far_field(n).reshape(-1, 3)
    b = focus.ideal_far_field(n, f_number=0.1).reshape(-1, 3)
    a = a / np.sqrt(np.sum(np.abs(a) ** 2))
    b = b / np.sqrt(np.sum(np.abs(b) ** 2))
    overlap = abs(np.vdot(a.ravel(), b.ravel()))
    assert overlap > 0.999


def test_compute_checks_catches_flipped_bc():
    Ec, Bc, grid, _ = _compute_small()
    checks = focus.compute_checks(Ec, lambda0_m=LAM, power_w=POWER, grid=grid, Bc=-Bc)
    assert checks["power_conservation_rel_err"] > 1.5


def test_gaussian_far_field_runs():
    n = _sample_n()
    E = focus.gaussian_far_field(n, f_number=F_NUMBER, w0_over_rmax=0.7, apod_order=8.0)
    assert E.shape == (*n.shape[:-1], 3)
    assert np.max(np.abs(E)) > 0.0


def test_comparison_reports_degradation():
    rep = comparison.summarize_degradation(f_number=F_NUMBER)
    assert rep["mse"] > 0.0
    assert rep["correlation"] < 1.0
    assert 0.0 < rep["transmission"] <= 1.0


def test_gaussian_sweep_lower_than_ideal():
    fns = [0.25, 1.0]
    ideal = sweep.kappa_eff_vs_f_number(
        fns, lambda0_m=1.0e-6, power_w=POWER, energy_gev=1.0, ntheta=128, nphi=256,
    )
    gauss = sweep.kappa_eff_vs_f_number(
        fns, lambda0_m=1.0e-6, power_w=POWER, energy_gev=1.0,
        far_field_builder=sweep.gaussian_far_field_builder(), ntheta=128, nphi=256,
    )
    # The realistic (degraded) beam yields a lower chi_max than the optimal baseline.
    assert gauss[0]["chi_max_per_GeV"] < ideal[0]["chi_max_per_GeV"]


def test_invalid_pol_rejected():
    with pytest.raises(ValueError, match="pol must be"):
        focus.bi_dipole_far_field(_sample_n(), pol="elliptical")
    params = cli.default_parameters()
    params["pol"] = "elliptical"
    with pytest.raises(ValueError, match="pol must be"):
        cli.generate_field_pack(params)


def test_circular_far_field_transverse_and_matches_mirror_reflection():
    n = _sample_n()
    a = focus.bi_dipole_far_field(n, pol="circular")
    assert np.max(np.abs(np.sum(n * a, axis=-1))) < 1e-12
    b = focus.ideal_far_field(n, f_number=F_NUMBER, pol="circular")
    a_flat = a.reshape(-1, 3)
    b_flat = b.reshape(-1, 3)
    a_flat = a_flat / np.sqrt(np.sum(np.abs(a_flat) ** 2))
    b_flat = b_flat / np.sqrt(np.sum(np.abs(b_flat) ** 2))
    assert abs(np.vdot(a_flat.ravel(), b_flat.ravel())) > 0.999


def test_circular_focal_field_amplitudes_phase_and_power():
    for mode in ("ideal", "gaussian"):
        ff_lp = (
            (lambda n: focus.bi_dipole_far_field(n, pol="z-linear"))
            if mode == "ideal"
            else (lambda n: focus.gaussian_far_field(n, f_number=F_NUMBER, pol="z-linear"))
        )
        ff_cp = (
            (lambda n: focus.bi_dipole_far_field(n, pol="circular"))
            if mode == "ideal"
            else (lambda n: focus.gaussian_far_field(n, f_number=F_NUMBER, pol="circular"))
        )
        ec_lp, bc_lp, grid = focus.focal_field_angular_spectrum(
            lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER,
            nx=24, ny=24, nz=24, span_xy_m=6.4e-6, span_y_m=3.2e-6,
            ntheta=24, nphi=48, far_field=ff_lp, pol="z-linear",
        )
        ec_cp, bc_cp, _ = focus.focal_field_angular_spectrum(
            lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER,
            nx=24, ny=24, nz=24, span_xy_m=6.4e-6, span_y_m=3.2e-6,
            ntheta=24, nphi=48, far_field=ff_cp, pol="circular",
        )
        chk_lp = focus.compute_checks(ec_lp, lambda0_m=LAM, power_w=POWER, grid=grid, Bc=bc_lp, pol="z-linear")
        chk_cp = focus.compute_checks(ec_cp, lambda0_m=LAM, power_w=POWER, grid=grid, Bc=bc_cp, pol="circular")

        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        assert np.isclose(chk_cp["a_max"] / chk_lp["a_max"], inv_sqrt2, rtol=1e-6)
        assert np.isclose(chk_cp["chi_max_per_GeV"] / chk_lp["chi_max_per_GeV"], inv_sqrt2, rtol=1e-6)
        assert 0.0 < chk_cp["power_conservation_rel_err"] < 0.10

        ic = 12
        ef = ec_cp[0, :, ic, ic, ic]
        bf = bc_cp[0, :, ic, ic, ic]
        assert ef[2].real > 0.0 and abs(ef[2].imag) < 1e-12 * ef[2].real
        assert np.isclose(ef[0], 1j * ef[2], rtol=1e-12)
        assert np.isclose(bf[0], -ef[2] / C.C_LIGHT, rtol=1e-12)
        assert np.isclose(bf[2], 1j * ef[2] / C.C_LIGHT, rtol=1e-12)

        # Circular polarization maintains constant |E(0, t)| across the optical cycle.
        phases = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        mags = [np.linalg.norm(np.real(ef * np.exp(1j * ph))) for ph in phases]
        assert np.allclose(mags, mags[0], rtol=1e-12)


def test_circular_sweep_scales_by_inv_sqrt2():
    fns = [0.25, 1.0]
    for builder_lp, builder_cp in (
        (sweep.ideal_far_field_builder("z-linear"), sweep.ideal_far_field_builder("circular")),
        (sweep.gaussian_far_field_builder(pol="z-linear"), sweep.gaussian_far_field_builder(pol="circular")),
    ):
        lp = sweep.kappa_eff_vs_f_number(fns, lambda0_m=1.0e-6, power_w=POWER, far_field_builder=builder_lp, ntheta=64, nphi=128)
        cp = sweep.kappa_eff_vs_f_number(fns, lambda0_m=1.0e-6, power_w=POWER, far_field_builder=builder_cp, ntheta=64, nphi=128)
        for row_lp, row_cp in zip(lp, cp):
            assert np.isclose(row_cp["a_max"] / row_lp["a_max"], 1.0 / np.sqrt(2.0), rtol=1e-6)
            assert np.isclose(row_cp["chi_max_per_GeV"] / row_lp["chi_max_per_GeV"], 1.0 / np.sqrt(2.0), rtol=1e-6)


def test_focal_field_slice_matches_3d_center_plane():
    Ec_3d, _, _ = focus.focal_field_angular_spectrum(
        lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER,
        nx=16, ny=16, nz=16, span_xy_m=3.2e-6, span_y_m=3.2e-6,
        ntheta=16, nphi=32,
    )
    Ec_slice, _, _, _ = focus.focal_field_slice(
        lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER,
        plane="y", coord_m=0.0, n1=16, n2=16,
        span1_m=3.2e-6, span2_m=3.2e-6, ntheta=16, nphi=32,
    )
    ec_3d_plane = Ec_3d[0, :, :, 8, :]
    assert np.allclose(np.abs(Ec_slice), np.abs(ec_3d_plane), rtol=1e-12)
    ref = Ec_slice[2, 8, 8]
    aligned_slice = Ec_slice * (np.conj(ref) / abs(ref))
    assert np.allclose(aligned_slice, ec_3d_plane, rtol=1e-12)


def test_build_metadata_extra_rejects_collision():
    _, _, grid, checks = _compute_small()
    with pytest.raises(ValueError, match="cannot overwrite core fields"):
        meta_mod.build_metadata(
            lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER, pol="z-linear",
            grid=grid, checks=checks, tau_fwhm_s=30.0e-15,
            extra={"pol": "circular"},
        )


def test_transmission_bounds_and_monotonicity():
    t_ap_narrow = comparison.aperture_transmission(0.5)
    t_ap_wide = comparison.aperture_transmission(0.9)
    assert 0.0 < t_ap_wide < t_ap_narrow <= 1.0
    t_apod = comparison.apodizer_transmission(0.7, 8.0)
    t_tot = comparison.total_transmission(0.7, 8.0)
    assert 0.0 < t_apod <= 1.0
    assert np.isclose(t_tot, comparison.aperture_transmission(0.7) * t_apod, rtol=1e-12)
    rep = comparison.summarize_degradation(f_number=F_NUMBER)
    assert np.isclose(rep["total_transmission"], t_tot, rtol=1e-12)


def test_kappa_sweep_folds_power_transmission():
    fns = [0.25, 1.0]
    t_val = sweep.gaussian_transmission(0.7, 8.0)
    base = sweep.kappa_eff_vs_f_number(
        fns, lambda0_m=1.0e-6, power_w=POWER,
        far_field_builder=sweep.gaussian_far_field_builder(), ntheta=64, nphi=128,
    )
    scaled_scalar = sweep.kappa_eff_vs_f_number(
        fns, lambda0_m=1.0e-6, power_w=POWER,
        far_field_builder=sweep.gaussian_far_field_builder(),
        power_transmission=t_val, ntheta=64, nphi=128,
    )
    scaled_fn = sweep.kappa_eff_vs_f_number(
        fns, lambda0_m=1.0e-6, power_w=POWER,
        far_field_builder=sweep.gaussian_far_field_builder(),
        power_transmission=lambda _fn: t_val, ntheta=64, nphi=128,
    )
    for r0, r1, r2 in zip(base, scaled_scalar, scaled_fn):
        assert np.isclose(r1["chi_max_per_GeV"], r0["chi_max_per_GeV"] * np.sqrt(t_val), rtol=1e-12)
        assert np.isclose(r1["a_max"], r0["a_max"] * np.sqrt(t_val), rtol=1e-12)
        assert np.isclose(r1["kappa_eff"], r0["kappa_eff"] * np.sqrt(t_val), rtol=1e-12)
        assert r1["transmission"] == pytest.approx(t_val)
        assert r2 == pytest.approx(r1)

    with pytest.raises(ValueError, match="power_transmission"):
        sweep.kappa_eff_vs_f_number([0.25], power_transmission=0.0)
    with pytest.raises(ValueError, match="power_transmission"):
        sweep.kappa_eff_vs_f_number([0.25], power_transmission=1.2)


def test_cli_account_for_losses_scales_focal_power_and_kappa():
    base_params = {**cli.default_parameters(), "nx": 16, "ny": 16, "nz": 16, "ntheta": 16, "nphi": 32}
    lossless = cli.generate_field_pack({**base_params, "account_for_losses": False})
    lossy = cli.generate_field_pack({**base_params, "account_for_losses": True})
    t_tot = lossless["metadata"]["degradation"]["total_transmission"]
    assert np.isclose(lossy["metadata"]["power_W"], POWER * t_tot, rtol=1e-12)
    assert np.isclose(lossy["metadata"]["checks"]["a_max"], lossless["metadata"]["checks"]["a_max"] * np.sqrt(t_tot), rtol=1e-12)
    assert 0.0 < lossy["metadata"]["checks"]["power_conservation_rel_err"] < 0.10
    assert np.isclose(
        lossy["metadata"]["degradation"]["kappa_eff_effective"],
        lossy["metadata"]["degradation"]["kappa_eff_focal"] * np.sqrt(t_tot),
        rtol=1e-12,
    )


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()