"""Command-line entrypoint and high-level API for the optics simulation.

Generates the bi-dipole focal-field pack (``Ec``, ``Bc``, ``metadata``) and writes:

* ``data/f{fnumber}.npz``  -- the field pack (gitignored), validated metadata;
* ``outs.json``            -- the metadata as JSON (the format sketched in ``outs.md``);
* the same JSON to stdout.

Default parameters reproduce the paper's numerical analysis (Fig. 5): ``lambda = 0.8 um``,
a 5-cycle Gaussian pulse (FWHM intensity), ``P = 1 PW``, f-number ``0.25`` and
``eps = 1 GeV``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from . import focus
from . import metadata as meta_mod

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_parameters() -> dict:
    """Return the default (paper-replicating) parameter set."""
    return {
        "lambda0_m": 0.8e-6,          # Ti:Sapphire, 800 nm
        "f_number": 0.25,             # full -y hemisphere (opening angle 90 deg)
        "power_w": 1.0e15,            # 1 PW
        "pol": "z-linear",            # bi-dipole is linearly polarized along z
        "n_cycles": 5.0,              # 5-cycle pulse (FWHM intensity)
        "energy_gev": 1.0,            # reference electron energy for chi
        "nx": 64, "ny": 64, "nz": 64,
        "span_xy_m": 4.0e-6,
        "span_y_m": 4.0e-6,
        "ntheta": 32,
        "nphi": 64,
        "focal_length_m": None,
    }


def generate_field_pack(params: dict) -> dict:
    """Compute the focal field pack and return the (validated) metadata JSON dict."""
    lambda0_m = params["lambda0_m"]
    f_number = params["f_number"]
    power_w = params["power_w"]

    Ec, Bc, grid = focus.focal_field_angular_spectrum(
        lambda0_m=lambda0_m,
        f_number=f_number,
        power_w=power_w,
        nx=params["nx"],
        ny=params["ny"],
        nz=params["nz"],
        span_xy_m=params["span_xy_m"],
        span_y_m=params["span_y_m"],
        ntheta=params["ntheta"],
        nphi=params["nphi"],
    )
    checks = focus.compute_checks(
        Ec, lambda0_m=lambda0_m, power_w=power_w, grid=grid, Bc=Bc
    )
    metadata = meta_mod.build_metadata(
        lambda0_m=lambda0_m,
        f_number=f_number,
        power_w=power_w,
        pol=params["pol"],
        grid=grid,
        checks=checks,
        n_cycles=params["n_cycles"],
        focal_length_m=params["focal_length_m"],
    )
    return {"Ec": Ec, "Bc": Bc, "metadata": metadata}


def save_field_pack(pack: dict, *, out_dir: Path | None = None) -> Path:
    """Write the ``.npz`` field pack and return its path."""
    out_dir = out_dir or (REPO_ROOT / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = pack["metadata"]["f_number"]
    path = out_dir / f"f{fn}.npz"
    np.savez_compressed(
        path,
        Ec=pack["Ec"],
        Bc=pack["Bc"],
        metadata=json.dumps(pack["metadata"]),
    )
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Bi-dipole optical simulation (Olofsson & Gonoskov, PRA 106, 063512)."
    )
    parser.add_argument("--lambda0-m", type=float, default=0.8e-6)
    parser.add_argument("--f-number", type=float, default=0.25)
    parser.add_argument("--power-w", type=float, default=1.0e15)
    parser.add_argument("--pol", choices=["z-linear", "circular"], default="z-linear")
    parser.add_argument("--n-cycles", type=float, default=5.0)
    parser.add_argument("--energy-gev", type=float, default=1.0)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--span-xy-m", type=float, default=4.0e-6)
    parser.add_argument("--span-y-m", type=float, default=4.0e-6)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nphi", type=int, default=64)
    parser.add_argument("--focal-length-m", type=float, default=None)
    parser.add_argument("--out-json", default=str(REPO_ROOT / "outs.json"))
    args = parser.parse_args(argv)

    params = {
        "lambda0_m": args.lambda0_m,
        "f_number": args.f_number,
        "power_w": args.power_w,
        "pol": args.pol,
        "n_cycles": args.n_cycles,
        "energy_gev": args.energy_gev,
        "nx": args.nx, "ny": args.ny, "nz": args.nz,
        "span_xy_m": args.span_xy_m,
        "span_y_m": args.span_y_m,
        "ntheta": args.ntheta,
        "nphi": args.nphi,
        "focal_length_m": args.focal_length_m,
    }

    pack = generate_field_pack(params)
    npz_path = save_field_pack(pack)

    meta_json = json.dumps(pack["metadata"], indent=2)
    out_json = Path(args.out_json)
    out_json.write_text(meta_json + "\n", encoding="utf-8")
    print(meta_json)

    print(f"\n[optics] field pack -> {npz_path}")
    print(f"[optics] metadata   -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
