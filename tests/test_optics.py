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


def test_circular_pol_rejected_loudly():
    params = cli.default_parameters()
    params["pol"] = "circular"
    with pytest.raises(NotImplementedError, match="z-linear"):
        cli.generate_field_pack(params)


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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()