"""Integration and contract tests between src.optics and src.particles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optics import cli
from src.particles import (
    compute_sigma_and_eta,
    e_field_to_a0,
    evaluate_eb,
    load_field_pack,
    propagate_ensemble,
    sample_electron_batch,
    simulate_qed_ensemble,
)

SCHEMA_PATH = REPO_ROOT / "data" / "schema.json"


def _make_small_optics_pack(mode: str = "gaussian", pol: str = "z-linear") -> dict:
    """Generate a compact optics focal-field pack for integration testing."""
    params = cli.default_parameters()
    params.update(
        mode=mode,
        pol=pol,
        nx=32,
        ny=32,
        nz=32,
        span_xy_m=6.4e-6,
        span_y_m=3.2e-6,
        ntheta=24,
        nphi=48,
    )
    return cli.generate_field_pack(params)


def test_optics_npz_matches_particles_schema_and_contract(tmp_path: Path) -> None:
    """Verify optics .npz output structure, dtypes, finiteness, schema, and power check."""
    pack = _make_small_optics_pack()
    npz_path = cli.save_field_pack(pack, out_dir=tmp_path)

    with np.load(npz_path, allow_pickle=False) as archive:
        assert {"Ec", "Bc", "metadata"}.issubset(archive.files)
        ec = archive["Ec"]
        bc = archive["Bc"]
        assert ec.shape == (1, 3, 32, 32, 32)
        assert bc.shape == (1, 3, 32, 32, 32)
        assert ec.dtype == np.complex128
        assert bc.dtype == np.complex128
        assert np.all(np.isfinite(ec))
        assert np.all(np.isfinite(bc))

        raw_meta = archive["metadata"].item()
        if isinstance(raw_meta, (bytes, bytearray)):
            raw_meta = raw_meta.decode("utf-8")
        metadata = json.loads(raw_meta)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=metadata, schema=schema)

    power_rel_err = metadata["checks"]["power_conservation_rel_err"]
    assert 0.0 < power_rel_err < 0.10


@pytest.mark.parametrize("mode", ["gaussian", "ideal"])
@pytest.mark.parametrize("pol", ["z-linear", "circular"])
def test_raw_optics_pack_loads_in_particles(tmp_path: Path, mode: str, pol: str) -> None:
    """Verify unmodified optics packs load directly via load_field_pack."""
    pack = _make_small_optics_pack(mode=mode, pol=pol)
    npz_path = cli.save_field_pack(pack, out_dir=tmp_path)
    loaded = load_field_pack(npz_path)
    assert loaded.pol == pol


@pytest.mark.parametrize("mode", ["gaussian", "ideal"])
@pytest.mark.parametrize("pol", ["z-linear", "circular"])
def test_optics_pack_propagates_in_particles(tmp_path: Path, mode: str, pol: str) -> None:
    """Verify optics pack loads, evaluates, and pushes electrons cleanly."""
    pack = _make_small_optics_pack(mode=mode, pol=pol)
    npz_path = cli.save_field_pack(pack, out_dir=tmp_path)
    loaded = load_field_pack(npz_path)

    e_vec, b_vec = evaluate_eb(loaded, np.zeros(3), 0.0)
    assert e_vec[2] > 0.0
    assert b_vec[0] < 0.0

    a0_eval = e_field_to_a0(float(np.linalg.norm(e_vec)), loaded.freqs[0])
    assert np.isclose(a0_eval, loaded.checks.a_max, rtol=1e-2)

    batch = sample_electron_batch(
        n_particles=4,
        energy_gev=1.0,
        b_max_m=0.5 * loaded.lambda0_m,
        y0_m=loaded.grid.y0_m,
        rng=np.random.default_rng(0),
    )
    traj = propagate_ensemble(loaded, batch, method="vay")

    assert np.all(np.isfinite(traj.r_m))
    assert np.all(np.isfinite(traj.p_kgms))
    assert np.all(np.isfinite(traj.gamma))
    assert np.all(np.isfinite(traj.chi))
    assert float(np.max(traj.chi)) > 0.0


def test_circular_optics_pack_qed_selection_cuts(tmp_path: Path) -> None:
    """Verify CP optics pack drives QED simulation and CP selection cuts cleanly."""
    pack = _make_small_optics_pack(mode="gaussian", pol="circular")
    npz_path = cli.save_field_pack(pack, out_dir=tmp_path)
    loaded = load_field_pack(npz_path)

    batch = sample_electron_batch(
        n_particles=16,
        energy_gev=10.0,
        b_max_m=1.0 * loaded.lambda0_m,
        y0_m=loaded.grid.y0_m,
        rng=np.random.default_rng(7),
    )
    res = simulate_qed_ensemble(loaded, batch, mode="weighted", method="vay")
    obs = compute_sigma_and_eta(
        res.photons,
        pol=loaded.pol,
        a_max=loaded.checks.a_max,
        chi_max=loaded.checks.chi_max_per_GeV * batch.energy_gev,
        b_max_m=batch.b_max_m,
        n_electrons=batch.r0_m.shape[0],
    )
    assert res.photons.weight.size > 0
    assert np.isfinite(obs["sigma_cm2"]) and obs["sigma_cm2"] >= 0.0
    assert np.isfinite(obs["eta"]) and 0.0 <= obs["eta"] <= 1.0
