"""Integration and contract tests between src.optics and src.particles."""

from __future__ import annotations

import copy
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
    e_field_to_a0,
    evaluate_eb,
    load_field_pack,
    propagate_ensemble,
    sample_electron_batch,
)

SCHEMA_PATH = REPO_ROOT / "data" / "schema.json"


def _make_small_optics_pack() -> dict:
    """Generate a compact optics focal-field pack for integration testing."""
    params = cli.default_parameters()
    params.update(
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


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "Pending phase decision: focus.py -1j Gouy phase makes Re[Ec(0)]=0 at "
        "t_ref=0, failing load_field_pack a_max self-check"
    ),
)
def test_raw_optics_pack_loads_in_particles(tmp_path: Path) -> None:
    """Verify raw optics pack triggers expected a_max self-check failure on load."""
    pack = _make_small_optics_pack()
    npz_path = cli.save_field_pack(pack, out_dir=tmp_path)
    load_field_pack(npz_path)


def test_phase_aligned_optics_pack_propagates_in_particles(tmp_path: Path) -> None:
    """Verify phase-aligned optics pack loads, evaluates, and pushes electrons cleanly."""
    pack = _make_small_optics_pack()
    aligned_pack = {
        "Ec": pack["Ec"] * (-1j),
        "Bc": pack["Bc"] * (-1j),
        "metadata": copy.deepcopy(pack["metadata"]),
    }

    npz_path = cli.save_field_pack(aligned_pack, out_dir=tmp_path)
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
