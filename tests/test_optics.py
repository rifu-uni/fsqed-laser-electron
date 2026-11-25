"""Tests for the optics module (bi-dipole focusing simulation).

Run with ``pytest -q`` (the project's test task) or directly with
``python tests/test_optics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optics import constants as C  # noqa: E402
from optics import focus  # noqa: E402
from optics import metadata as meta_mod  # noqa: E402
from optics import sweep  # noqa: E402

LAM = 0.8e-6
F_NUMBER = 0.25
POWER = 1.0e15


def _compute_small():
    Ec, Bc, grid = focus.focal_field_angular_spectrum(
        lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER,
        nx=32, ny=32, nz=32, span_xy_m=3.2e-6, span_y_m=3.2e-6,
        ntheta=40, nphi=80,
    )
    checks = focus.compute_checks(Ec, lambda0_m=LAM, power_w=POWER, grid=grid, Bc=Bc)
    return Ec, Bc, grid, checks


def test_metadata_validates_against_schema():
    Ec, Bc, grid, checks = _compute_small()
    meta = meta_mod.build_metadata(
        lambda0_m=LAM, f_number=F_NUMBER, power_w=POWER, pol="z-linear",
        grid=grid, checks=checks, n_cycles=5.0,
    )
    meta_mod.validate(meta)  # raises on failure


def test_power_conservation():
    _, _, _, checks = _compute_small()
    assert checks["power_conservation_rel_err"] < 1e-6


def test_a_max_close_to_expected():
    _, _, _, checks = _compute_small()
    # The bi-dipole peak a0 is ~550 * sqrt(P/1PW) (lambda-independent); the discrete
    # Debye sum reproduces it to within ~10%.
    assert 0.85 * checks["a_max_expected"] < checks["a_max"] < 1.15 * checks["a_max_expected"]
    assert 450.0 < checks["a_max"] < 600.0


def test_chi_max_matches_a_max_scaling():
    _, _, _, checks = _compute_small()
    # chi ~ 2 a0 gamma (hbar omega / m c^2) per GeV, head-on.
    omega = C.omega_from_lambda(LAM)
    hbar_omega_over_mc2 = C.HBAR * omega / C.MC2_J
    expected = 2.0 * checks["a_max"] * C.GAMMA_PER_GEV * hbar_omega_over_mc2
    assert abs(checks["chi_max_per_GeV"] - expected) / expected < 0.05


def test_peak_location_at_focus():
    _, _, _, checks = _compute_small()
    x, y, z = checks["max_location_m"]
    assert abs(x) < 1e-7 and abs(y) < 1e-7 and abs(z) < 1e-7


def test_kappa_sweep_saturates_and_decreases():
    # Fig. 4 uses lambda = 1 um: chi_max saturates near kappa ~ 5.25 for f <= 0.25
    # and decreases for larger f-numbers.
    fns = [0.25, 0.5, 1.0, 2.0]
    res = sweep.kappa_eff_vs_f_number(
        fns, lambda0_m=1.0e-6, power_w=POWER, energy_gev=1.0,
        ntheta=128, nphi=256,
    )
    chi = [r["chi_max_per_GeV"] for r in res]
    # saturating value close to the paper's 5.25 (within ~10%)
    assert abs(chi[0] - 5.25) / 5.25 < 0.15
    # monotonically decreasing with f-number
    assert chi[0] > chi[1] > chi[2] > chi[3]


def test_far_field_is_transverse():
    # The bi-dipole far field must be transverse: n . E = 0.
    rng = np.random.default_rng(0)
    n = rng.normal(size=(200, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    E = focus.bi_dipole_far_field(n)
    dot = np.sum(n * E, axis=-1)
    assert np.max(np.abs(dot)) < 1e-12


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
