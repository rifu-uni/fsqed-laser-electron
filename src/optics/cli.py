"""Command-line entrypoint and high-level API for the optics simulation.

Generates the focal-field pack (``Ec``, ``Bc``, ``metadata``) and writes the metadata
JSON (the format sketched in ``outs.md``).

Two execution modes are supported:

* ``--mode ideal``    : analytic Sheppard-Larkin far field (eqs. 9-10) -- the
                        theoretical baseline;
* ``--mode gaussian`` : realistic optical pipeline -- a Gaussian laser beam shaped by
                        an apodizer and reflected off the parabolic mirror (default).

The laser parameters target the FSQED regime and are configurable: Ti:Sapphire
``lambda = 0.8 um``, a Gaussian pulse of tens of fs FWHM and PW-class peak power.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import comparison
from . import constants as C
from . import focus
from . import metadata as meta_mod

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_parameters() -> dict:
    """Return the default parameter set (FSQED laser, realistic optical pipeline)."""
    return {
        "lambda0_m": 0.8e-6,          # Ti:Sapphire, 800 nm
        "f_number": 0.25,             # full -y hemisphere (opening angle 90 deg)
        "power_w": 1.0e15,            # 1 PW peak power
        "pol": "z-linear",
        "tau_fwhm_s": 30.0e-15,       # 30 fs pulse (FWHM intensity), FSQED regime
        "energy_gev": 1.0,
        "mode": "gaussian",           # realistic optical pipeline
        "w0_over_rmax": 0.7,          # Gaussian waist / mirror radius
        "apod_order": 8.0,            # super-Gaussian apodizer order
        "nx": 64, "ny": 64, "nz": 64,
        "span_xy_m": 6.4e-6,
        "span_y_m": 4.0e-6,
        "ntheta": 32,
        "nphi": 64,
        "focal_length_m": None,
        "account_for_losses": False,
    }


def build_far_field(params: dict):
    """Return the ``far_field(n)`` callable for the chosen execution mode."""
    fn = params["f_number"]
    pol = focus._validate_pol(params["pol"])
    if params["mode"] == "ideal":
        return lambda n: focus.bi_dipole_far_field(n, pol=pol)
    return lambda n: focus.gaussian_far_field(
        n, f_number=fn,
        w0_over_rmax=params["w0_over_rmax"],
        apod_order=params["apod_order"],
        pol=pol,
    )


def generate_field_pack(params: dict) -> dict:
    """Compute the focal field pack and return the (validated) metadata dict."""
    pol = focus._validate_pol(params["pol"])

    lambda0_m = params["lambda0_m"]
    f_number = params["f_number"]
    source_power_w = float(params["power_w"])
    n_cycles = params.get("n_cycles")
    tau_fwhm_s = params.get("tau_fwhm_s")
    if tau_fwhm_s is None and n_cycles is not None:
        tau_fwhm_s = C.tau_fwhm_from_cycles(n_cycles, lambda0_m)

    account_for_losses = bool(params.get("account_for_losses", False))
    degradation = None
    focal_power_w = source_power_w
    if params["mode"] == "gaussian":
        degradation = comparison.summarize_degradation(
            f_number=f_number,
            w0_over_rmax=params["w0_over_rmax"],
            apod_order=params["apod_order"],
        )
        if account_for_losses:
            focal_power_w = source_power_w * float(degradation["total_transmission"])

    far_field = build_far_field(params)
    Ec, Bc, grid = focus.focal_field_angular_spectrum(
        lambda0_m=lambda0_m, f_number=f_number, power_w=focal_power_w,
        nx=params["nx"], ny=params["ny"], nz=params["nz"],
        span_xy_m=params["span_xy_m"], span_y_m=params["span_y_m"],
        ntheta=params["ntheta"], nphi=params["nphi"], far_field=far_field,
        pol=pol,
    )
    checks = focus.compute_checks(
        Ec, lambda0_m=lambda0_m, power_w=focal_power_w, grid=grid, Bc=Bc, pol=pol
    )
    extra: dict = {"mode": params["mode"]}
    if params["mode"] == "gaussian":
        assert degradation is not None
        scale_focal = float(np.sqrt(focal_power_w / 1.0e15) * (1.0e-6 / lambda0_m))
        kappa_eff_focal = float(checks["chi_max_per_GeV"] / scale_focal)
        total_t = float(degradation["total_transmission"])
        degradation["kappa_eff_focal"] = kappa_eff_focal
        degradation["kappa_eff_effective"] = float(kappa_eff_focal * np.sqrt(total_t))
        degradation["source_power_W"] = source_power_w
        degradation["focal_power_W"] = focal_power_w
        degradation["account_for_losses"] = account_for_losses
        extra["w0_over_rmax"] = params["w0_over_rmax"]
        extra["apod_order"] = params["apod_order"]
        extra["degradation"] = degradation
    metadata = meta_mod.build_metadata(
        lambda0_m=lambda0_m, f_number=f_number, power_w=focal_power_w,
        pol=params["pol"], grid=grid, checks=checks,
        tau_fwhm_s=tau_fwhm_s, n_cycles=n_cycles,
        focal_length_m=params["focal_length_m"],
        extra=extra,
    )
    return {"Ec": Ec, "Bc": Bc, "metadata": metadata}


def save_field_pack(pack: dict, *, out_dir: Path | None = None) -> Path:
    """Write the ``.npz`` field pack and return its path."""
    out_dir = out_dir or (REPO_ROOT / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = pack["metadata"]["f_number"]
    path = out_dir / f"f{fn}.npz"
    np.savez_compressed(
        path, Ec=pack["Ec"], Bc=pack["Bc"], metadata=json.dumps(pack["metadata"])
    )
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Bi-dipole optical simulation (Olofsson & Gonoskov, PRA 106, 063512)."
    )
    parser.add_argument("--mode", choices=["ideal", "gaussian"], default="gaussian")
    parser.add_argument("--lambda0-m", type=float, default=0.8e-6)
    parser.add_argument("--f-number", type=float, default=0.25)
    parser.add_argument("--power-w", type=float, default=1.0e15)
    parser.add_argument(
        "--pol",
        choices=["z-linear", "circular"],
        default="z-linear",
        help="Polarization mode: 'z-linear' (LP) or 'circular' (CP).",
    )
    parser.add_argument("--tau-fwhm-s", type=float, default=30.0e-15)
    parser.add_argument("--n-cycles", type=float, default=None, help="Optional pulse duration in optical cycles (overrides --tau-fwhm-s).")
    parser.add_argument("--energy-gev", type=float, default=1.0)
    parser.add_argument("--w0-over-rmax", type=float, default=0.7)
    parser.add_argument("--apod-order", type=float, default=8.0)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--span-xy-m", type=float, default=6.4e-6)
    parser.add_argument("--span-y-m", type=float, default=4.0e-6)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nphi", type=int, default=64)
    parser.add_argument("--focal-length-m", type=float, default=None)
    parser.add_argument(
        "--account-for-losses",
        action="store_true",
        help="Scale focal power in gaussian mode by aperture*apodizer total transmission.",
    )
    parser.add_argument("--out-json", default=str(REPO_ROOT / "outs.json"))
    args = parser.parse_args(argv)

    tau_fwhm_s = (
        C.tau_fwhm_from_cycles(args.n_cycles, args.lambda0_m)
        if args.n_cycles is not None
        else args.tau_fwhm_s
    )
    params = {
        "lambda0_m": args.lambda0_m,
        "f_number": args.f_number,
        "power_w": args.power_w,
        "pol": args.pol,
        "tau_fwhm_s": tau_fwhm_s,
        "n_cycles": args.n_cycles,
        "energy_gev": args.energy_gev,
        "mode": args.mode,
        "w0_over_rmax": args.w0_over_rmax,
        "apod_order": args.apod_order,
        "nx": args.nx, "ny": args.ny, "nz": args.nz,
        "span_xy_m": args.span_xy_m,
        "span_y_m": args.span_y_m,
        "ntheta": args.ntheta,
        "nphi": args.nphi,
        "focal_length_m": args.focal_length_m,
        "account_for_losses": args.account_for_losses,
    }

    pack = generate_field_pack(params)
    npz_path = save_field_pack(pack)

    meta_json = json.dumps(pack["metadata"], indent=2)
    Path(args.out_json).write_text(meta_json + "\n", encoding="utf-8")
    print(meta_json)
    print(f"\n[optics] mode      -> {params['mode']}")
    print(f"[optics] field pack -> {npz_path}")
    print(f"[optics] metadata   -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())