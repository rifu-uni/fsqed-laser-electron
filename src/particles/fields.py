"""Field pack boundary loader, Maxwell curl reconstruction, Numba interpolator, and analytic bidipole generator."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from numba import njit
from scipy.special import spherical_jn

from .constants import (
    A_BIDIPOLE_1PW_1UM,
    C_LIGHT,
    E_CRIT,
    FWHM_TO_FIELD_SIGMA,
    KAPPA_BIDIPOLE_MAX,
    e_field_to_a0,
    energy_gev_to_gamma,
    omega_from_lambda,
    period_from_lambda,
    tau_fwhm_to_sigma_t,
    theoretical_chi_max,
)

_DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schema.json"


@dataclass(frozen=True)
class GridSpec:
    """Uniform 3D Cartesian spatial grid specification (meters)."""

    x0_m: float
    dx_m: float
    Nx: int
    y0_m: float
    dy_m: float
    Ny: int
    z0_m: float
    dz_m: float
    Nz: int

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.x0_m)
            and math.isfinite(self.y0_m)
            and math.isfinite(self.z0_m)
            and math.isfinite(self.dx_m)
            and math.isfinite(self.dy_m)
            and math.isfinite(self.dz_m)
        ):
            raise ValueError("Grid origin and step values must be finite")
        if self.dx_m <= 0.0 or self.dy_m <= 0.0 or self.dz_m <= 0.0:
            raise ValueError("Grid spacings dx_m, dy_m, dz_m must be positive")
        if self.Nx < 2 or self.Ny < 2 or self.Nz < 2:
            raise ValueError("Grid counts Nx, Ny, Nz must be at least 2")

    @property
    def x_axis(self) -> np.ndarray:
        return self.x0_m + np.arange(self.Nx, dtype=np.float64) * self.dx_m

    @property
    def y_axis(self) -> np.ndarray:
        return self.y0_m + np.arange(self.Ny, dtype=np.float64) * self.dy_m

    @property
    def z_axis(self) -> np.ndarray:
        return self.z0_m + np.arange(self.Nz, dtype=np.float64) * self.dz_m


@dataclass(frozen=True)
class PulseSpec:
    """Temporal pulse envelope metadata (intensity FWHM convention)."""

    model: str
    convention: str
    tau_fwhm_s: float
    sigma_t_s: float
    t_ref_s: float = 0.0


@dataclass(frozen=True)
class FieldChecks:
    """Self-validation checks stored inside field pack metadata."""

    a_max: float
    a_max_expected: float
    chi_max_per_GeV: float
    power_conservation_rel_err: float
    max_location_m: tuple[float, float, float]


@dataclass(frozen=True)
class FieldPack:
    """Validated electromagnetic phasor field pack in SI units."""

    Ec: np.ndarray  # (Nf, 3, Nz, Ny, Nx) complex128, V/m
    Bc: np.ndarray  # (Nf, 3, Nz, Ny, Nx) complex128, T
    grid: GridSpec
    freqs: np.ndarray  # (Nf,) float64, rad/s
    power_W: float
    f_number: float
    lambda0_m: float
    pol: str  # "z-linear" or "circular"
    pulse: PulseSpec
    checks: FieldChecks
    metadata_raw: dict[str, Any]


def _load_schema(schema_path: str | Path | None) -> dict[str, Any]:
    path = _DEFAULT_SCHEMA_PATH if schema_path is None else Path(schema_path)
    if not path.is_file():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _decode_metadata(raw_meta: Any) -> dict[str, Any]:
    if isinstance(raw_meta, np.ndarray):
        if raw_meta.ndim == 0:
            raw_meta = raw_meta.item()
        elif raw_meta.size == 1:
            raw_meta = raw_meta.reshape(()).item()
        else:
            raise ValueError(f"metadata array must be scalar, got shape {raw_meta.shape}")
    if isinstance(raw_meta, (bytes, bytearray, np.bytes_)):
        raw_meta = bytes(raw_meta).decode("utf-8")
    if not isinstance(raw_meta, str):
        raise ValueError(f"metadata must decode to JSON string, got {type(raw_meta).__name__}")
    parsed = json.loads(raw_meta)
    if not isinstance(parsed, dict):
        raise ValueError("Decoded metadata JSON must be an object")
    return parsed


def _parse_pulse_spec(pulse_raw: dict[str, Any]) -> PulseSpec:
    model = str(pulse_raw["model"])
    convention = str(pulse_raw["convention"])
    t_ref_s = float(pulse_raw.get("t_ref_s", 0.0))
    if not math.isfinite(t_ref_s):
        raise ValueError(f"pulse.t_ref_s must be finite, got {t_ref_s}")

    has_tau = "tau_fwhm_s" in pulse_raw
    has_sigma = "sigma_t_s" in pulse_raw

    if model == "gauss":
        if not has_tau and not has_sigma:
            raise ValueError("pulse.model='gauss' requires tau_fwhm_s or sigma_t_s")
        if has_tau and has_sigma:
            tau_fwhm_s = float(pulse_raw["tau_fwhm_s"])
            sigma_t_s = float(pulse_raw["sigma_t_s"])
            expected_sigma = tau_fwhm_to_sigma_t(tau_fwhm_s)
            if not math.isclose(sigma_t_s, expected_sigma, rel_tol=1.0e-3, abs_tol=0.0):
                raise ValueError(
                    f"Inconsistent pulse.tau_fwhm_s ({tau_fwhm_s}) and pulse.sigma_t_s ({sigma_t_s}); "
                    f"expected sigma_t_s={expected_sigma}"
                )
        elif has_tau:
            tau_fwhm_s = float(pulse_raw["tau_fwhm_s"])
            sigma_t_s = tau_fwhm_to_sigma_t(tau_fwhm_s)
        else:
            sigma_t_s = float(pulse_raw["sigma_t_s"])
            if not math.isfinite(sigma_t_s) or sigma_t_s <= 0.0:
                raise ValueError(f"pulse.sigma_t_s must be positive, got {sigma_t_s}")
            tau_fwhm_s = sigma_t_s * FWHM_TO_FIELD_SIGMA
        return PulseSpec(
            model=model,
            convention=convention,
            tau_fwhm_s=tau_fwhm_s,
            sigma_t_s=sigma_t_s,
            t_ref_s=t_ref_s,
        )

    tau_fwhm_s = float(pulse_raw.get("tau_fwhm_s", 0.0))
    sigma_t_s = float(pulse_raw.get("sigma_t_s", 0.0))
    return PulseSpec(
        model=model,
        convention=convention,
        tau_fwhm_s=tau_fwhm_s,
        sigma_t_s=sigma_t_s,
        t_ref_s=t_ref_s,
    )


def compute_bc_from_ec_curl(
    Ec: np.ndarray,
    freqs: np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    """Reconstruct magnetic field phasors Bc = (i / omega_k) * curl(Ec) via central differences."""
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    if freqs_arr.ndim != 1 or freqs_arr.size == 0:
        raise ValueError(f"freqs must be a non-empty 1D array, got shape {freqs_arr.shape}")
    if not np.all(np.isfinite(freqs_arr)) or np.any(freqs_arr <= 0.0):
        raise ValueError("freqs must contain finite positive angular frequencies")

    ec_arr = np.asarray(Ec, dtype=np.complex128)
    expected_shape = (freqs_arr.size, 3, grid.Nz, grid.Ny, grid.Nx)
    if ec_arr.shape != expected_shape:
        raise ValueError(f"Ec shape {ec_arr.shape} does not match expected {expected_shape}")

    ex = ec_arr[:, 0, :, :, :]
    ey = ec_arr[:, 1, :, :, :]
    ez = ec_arr[:, 2, :, :, :]

    dez_dy = np.gradient(ez, grid.dy_m, axis=-2)
    dey_dz = np.gradient(ey, grid.dz_m, axis=-3)
    dex_dz = np.gradient(ex, grid.dz_m, axis=-3)
    dez_dx = np.gradient(ez, grid.dx_m, axis=-1)
    dey_dx = np.gradient(ey, grid.dx_m, axis=-1)
    dex_dy = np.gradient(ex, grid.dy_m, axis=-2)

    factor = (1.0j / freqs_arr)[:, None, None, None]
    bc = np.empty_like(ec_arr, dtype=np.complex128)
    bc[:, 0, :, :, :] = factor * (dez_dy - dey_dz)
    bc[:, 1, :, :, :] = factor * (dex_dz - dez_dx)
    bc[:, 2, :, :, :] = factor * (dey_dx - dex_dy)
    return np.ascontiguousarray(bc, dtype=np.complex128)


def _normalize_ec_array(
    ec_raw: np.ndarray,
    n_freqs: int,
    grid: GridSpec,
    pol: str,
) -> np.ndarray:
    ec_arr = np.asarray(ec_raw)
    if ec_arr.ndim == 4:
        if pol != "z-linear":
            raise ValueError(
                f"Legacy 4D Ec scalar Ez arrays are only supported for pol='z-linear', got pol={pol!r}"
            )
        expected_4d = (n_freqs, grid.Nz, grid.Ny, grid.Nx)
        if ec_arr.shape != expected_4d:
            raise ValueError(f"4D Ec shape {ec_arr.shape} does not match {expected_4d}")
        ec_5d = np.zeros((n_freqs, 3, grid.Nz, grid.Ny, grid.Nx), dtype=np.complex128)
        ec_5d[:, 2, :, :, :] = ec_arr
    elif ec_arr.ndim == 5:
        expected_5d = (n_freqs, 3, grid.Nz, grid.Ny, grid.Nx)
        if ec_arr.shape != expected_5d:
            raise ValueError(f"5D Ec shape {ec_arr.shape} does not match {expected_5d}")
        ec_5d = np.ascontiguousarray(ec_arr, dtype=np.complex128)
    else:
        raise ValueError(f"Ec must be 4D or 5D, got {ec_arr.ndim}D with shape {ec_arr.shape}")

    if not np.all(np.isfinite(ec_5d.real)) or not np.all(np.isfinite(ec_5d.imag)):
        raise ValueError("Ec contains non-finite values")
    return ec_5d


def _compute_peak_e_at_t_ref(
    ec: np.ndarray,
    freqs: np.ndarray,
    t_ref_s: float,
) -> float:
    phases = np.exp(1.0j * freqs * t_ref_s)[:, None, None, None, None]
    e_ref = np.sum(np.real(ec * phases), axis=0)  # (3, Nz, Ny, Nx)
    e_mag = np.sqrt(e_ref[0] ** 2 + e_ref[1] ** 2 + e_ref[2] ** 2)
    return float(np.max(e_mag))


def load_field_pack(
    npz_path: str | Path,
    schema_path: str | Path | None = None,
    check_rtol: float = 0.05,
    max_power_rel_err: float = 0.10,
) -> FieldPack:
    """Load, schema-validate, and self-check an electromagnetic field pack (.npz)."""
    if not math.isfinite(check_rtol) or check_rtol <= 0.0:
        raise ValueError(f"check_rtol must be finite and positive, got {check_rtol}")
    if not math.isfinite(max_power_rel_err) or max_power_rel_err < 0.0:
        raise ValueError(f"max_power_rel_err must be finite and non-negative, got {max_power_rel_err}")

    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Field pack not found: {path}")

    schema = _load_schema(schema_path)

    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if "Ec" not in keys or "metadata" not in keys:
            missing = sorted({"Ec", "metadata"} - keys)
            raise ValueError(f"Field pack {path} is missing required keys: {missing}")

        metadata = _decode_metadata(archive["metadata"])
        try:
            jsonschema.validate(instance=metadata, schema=schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"Field pack metadata failed schema validation: {exc.message}") from exc

        freqs = np.asarray(metadata["freqs"], dtype=np.float64)
        if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0.0):
            raise ValueError("freqs must contain finite positive values")
        if freqs.size > 1 and not np.all(np.diff(freqs) > 0.0):
            raise ValueError("freqs must be strictly increasing")

        grid_raw = metadata["grid"]
        grid = GridSpec(
            x0_m=float(grid_raw["x0_m"]),
            dx_m=float(grid_raw["dx_m"]),
            Nx=int(grid_raw["Nx"]),
            y0_m=float(grid_raw["y0_m"]),
            dy_m=float(grid_raw["dy_m"]),
            Ny=int(grid_raw["Ny"]),
            z0_m=float(grid_raw["z0_m"]),
            dz_m=float(grid_raw["dz_m"]),
            Nz=int(grid_raw["Nz"]),
        )

        pol = str(metadata["pol"])
        power_w = float(metadata["power_W"])
        f_number = float(metadata["f_number"])
        lambda0_m = float(metadata["lambda0_m"])
        pulse = _parse_pulse_spec(metadata["pulse"])

        checks_raw = metadata["checks"]
        loc_raw = checks_raw["max_location_m"]
        checks = FieldChecks(
            a_max=float(checks_raw["a_max"]),
            a_max_expected=float(checks_raw["a_max_expected"]),
            chi_max_per_GeV=float(checks_raw["chi_max_per_GeV"]),
            power_conservation_rel_err=float(checks_raw["power_conservation_rel_err"]),
            max_location_m=(float(loc_raw[0]), float(loc_raw[1]), float(loc_raw[2])),
        )

        ec = _normalize_ec_array(archive["Ec"], freqs.size, grid, pol)
        if "Bc" in keys:
            bc = np.ascontiguousarray(archive["Bc"], dtype=np.complex128)
            if bc.shape != ec.shape:
                raise ValueError(f"Bc shape {bc.shape} does not match Ec shape {ec.shape}")
            if not np.all(np.isfinite(bc.real)) or not np.all(np.isfinite(bc.imag)):
                raise ValueError("Bc contains non-finite values")
        else:
            bc = compute_bc_from_ec_curl(ec, freqs, grid)

    omega0 = omega_from_lambda(lambda0_m)
    e_peak = _compute_peak_e_at_t_ref(ec, freqs, pulse.t_ref_s)
    a_max_loaded = e_field_to_a0(e_peak, omega0)
    rel_diff = abs(a_max_loaded - checks.a_max) / max(checks.a_max, 1.0e-30)
    if rel_diff > check_rtol:
        raise ValueError(
            f"a_max self-check failed: loaded a_max={a_max_loaded:.6g}, "
            f"metadata checks.a_max={checks.a_max:.6g} (rel_diff={rel_diff:.4g} > {check_rtol})"
        )
    if checks.power_conservation_rel_err > max_power_rel_err:
        raise ValueError(
            f"power_conservation_rel_err ({checks.power_conservation_rel_err:.4g}) "
            f"exceeds threshold ({max_power_rel_err:.4g})"
        )

    return FieldPack(
        Ec=ec,
        Bc=bc,
        grid=grid,
        freqs=freqs,
        power_W=power_w,
        f_number=f_number,
        lambda0_m=lambda0_m,
        pol=pol,
        pulse=pulse,
        checks=checks,
        metadata_raw=metadata,
    )


def save_field_pack(
    npz_path: str | Path,
    pack_or_ec: FieldPack | np.ndarray,
    metadata: dict[str, Any] | None = None,
    Bc: np.ndarray | None = None,
    schema_path: str | Path | None = None,
    include_bc: bool = True,
) -> None:
    """Validate and save a FieldPack or (Ec, metadata[, Bc]) to a compressed .npz archive."""
    if isinstance(pack_or_ec, FieldPack):
        ec = np.ascontiguousarray(pack_or_ec.Ec, dtype=np.complex128)
        bc_to_save = np.ascontiguousarray(pack_or_ec.Bc, dtype=np.complex128) if include_bc else None
        meta_dict = pack_or_ec.metadata_raw
    else:
        if metadata is None:
            raise ValueError("metadata must be provided when saving raw Ec array")
        ec = np.ascontiguousarray(pack_or_ec, dtype=np.complex128)
        bc_to_save = np.ascontiguousarray(Bc, dtype=np.complex128) if (Bc is not None and include_bc) else None
        meta_dict = metadata

    schema = _load_schema(schema_path)
    try:
        jsonschema.validate(instance=meta_dict, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Field pack metadata failed schema validation: {exc.message}") from exc

    path = Path(npz_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "Ec": ec,
        "metadata": json.dumps(meta_dict),
    }
    if bc_to_save is not None:
        payload["Bc"] = bc_to_save
    np.savez_compressed(path, **payload)


@njit(cache=True)
def evaluate_eb_numba(
    x_m: float,
    y_m: float,
    z_m: float,
    t_s: float,
    dt_jitter_s: float,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Trilinearly interpolate complex phasors and synthesize real E (V/m) and B (T) at (r, t)."""
    ux = (x_m - x0_m) / dx_m
    uy = (y_m - y0_m) / dy_m
    uz = (z_m - z0_m) / dz_m

    if not (0.0 <= ux <= Nx - 1 and 0.0 <= uy <= Ny - 1 and 0.0 <= uz <= Nz - 1):
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    ix0 = int(math.floor(ux))
    if ix0 >= Nx - 1:
        ix0 = Nx - 2
    iy0 = int(math.floor(uy))
    if iy0 >= Ny - 1:
        iy0 = Ny - 2
    iz0 = int(math.floor(uz))
    if iz0 >= Nz - 1:
        iz0 = Nz - 2

    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    wx1 = ux - ix0
    wy1 = uy - iy0
    wz1 = uz - iz0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    w000 = wz0 * wy0 * wx0
    w001 = wz0 * wy0 * wx1
    w010 = wz0 * wy1 * wx0
    w011 = wz0 * wy1 * wx1
    w100 = wz1 * wy0 * wx0
    w101 = wz1 * wy0 * wx1
    w110 = wz1 * wy1 * wx0
    w111 = wz1 * wy1 * wx1

    t_eff = t_s + dt_jitter_s
    env = 1.0
    if use_gauss_env:
        arg = (t_eff - t_ref_s) / sigma_t_s
        env = math.exp(-0.5 * arg * arg)

    e_out = np.zeros(3, dtype=np.float64)
    b_out = np.zeros(3, dtype=np.float64)
    n_freqs = freqs.shape[0]

    for k in range(n_freqs):
        phase = freqs[k] * t_eff
        cos_p = math.cos(phase)
        sin_p = math.sin(phase)
        for c in range(3):
            ec_val = (
                w000 * Ec[k, c, iz0, iy0, ix0]
                + w001 * Ec[k, c, iz0, iy0, ix1]
                + w010 * Ec[k, c, iz0, iy1, ix0]
                + w011 * Ec[k, c, iz0, iy1, ix1]
                + w100 * Ec[k, c, iz1, iy0, ix0]
                + w101 * Ec[k, c, iz1, iy0, ix1]
                + w110 * Ec[k, c, iz1, iy1, ix0]
                + w111 * Ec[k, c, iz1, iy1, ix1]
            )
            bc_val = (
                w000 * Bc[k, c, iz0, iy0, ix0]
                + w001 * Bc[k, c, iz0, iy0, ix1]
                + w010 * Bc[k, c, iz0, iy1, ix0]
                + w011 * Bc[k, c, iz0, iy1, ix1]
                + w100 * Bc[k, c, iz1, iy0, ix0]
                + w101 * Bc[k, c, iz1, iy0, ix1]
                + w110 * Bc[k, c, iz1, iy1, ix0]
                + w111 * Bc[k, c, iz1, iy1, ix1]
            )
            e_out[c] += ec_val.real * cos_p - ec_val.imag * sin_p
            b_out[c] += bc_val.real * cos_p - bc_val.imag * sin_p

    for c in range(3):
        e_out[c] *= env
        b_out[c] *= env

    return e_out, b_out


def evaluate_eb(
    pack: FieldPack,
    r_m: np.ndarray,
    t_s: float,
    dt_jitter_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate real electric (V/m) and magnetic (T) fields at a single 3D point and time."""
    r_arr = np.asarray(r_m, dtype=np.float64)
    if r_arr.shape != (3,):
        raise ValueError(f"r_m must have shape (3,), got {r_arr.shape}")
    if not (math.isfinite(t_s) and math.isfinite(dt_jitter_s)):
        raise ValueError("t_s and dt_jitter_s must be finite")

    use_gauss_env = (pack.freqs.size == 1) and (pack.pulse.model == "gauss")
    sigma_t = pack.pulse.sigma_t_s if use_gauss_env else 1.0
    return evaluate_eb_numba(
        float(r_arr[0]),
        float(r_arr[1]),
        float(r_arr[2]),
        float(t_s),
        float(dt_jitter_s),
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


@njit(cache=True)
def _evaluate_eb_batch_numba(
    r_m_batch: np.ndarray,
    t_s: float,
    dt_jitter_s: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray]:
    n_pts = r_m_batch.shape[0]
    e_batch = np.zeros((n_pts, 3), dtype=np.float64)
    b_batch = np.zeros((n_pts, 3), dtype=np.float64)
    for i in range(n_pts):
        e_i, b_i = evaluate_eb_numba(
            r_m_batch[i, 0],
            r_m_batch[i, 1],
            r_m_batch[i, 2],
            t_s,
            dt_jitter_s[i],
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
        for c in range(3):
            e_batch[i, c] = e_i[c]
            b_batch[i, c] = b_i[c]
    return e_batch, b_batch


def evaluate_eb_batch(
    pack: FieldPack,
    r_m_batch: np.ndarray,
    t_s: float,
    dt_jitter_s: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate real E (V/m) and B (T) fields at a batch of points with shape (N, 3)."""
    r_arr = np.asarray(r_m_batch, dtype=np.float64)
    if r_arr.ndim != 2 or r_arr.shape[1] != 3:
        raise ValueError(f"r_m_batch must have shape (N, 3), got {r_arr.shape}")
    if not math.isfinite(t_s):
        raise ValueError(f"t_s must be finite, got {t_s}")

    n_pts = r_arr.shape[0]
    if np.ndim(dt_jitter_s) == 0:
        jitter_scalar = float(dt_jitter_s)
        if not math.isfinite(jitter_scalar):
            raise ValueError(f"dt_jitter_s must be finite, got {dt_jitter_s}")
        jitter_arr = np.full(n_pts, jitter_scalar, dtype=np.float64)
    else:
        jitter_arr = np.asarray(dt_jitter_s, dtype=np.float64)
        if jitter_arr.shape != (n_pts,) or not np.all(np.isfinite(jitter_arr)):
            raise ValueError(f"dt_jitter_s must be finite with shape ({n_pts},)")

    use_gauss_env = (pack.freqs.size == 1) and (pack.pulse.model == "gauss")
    sigma_t = pack.pulse.sigma_t_s if use_gauss_env else 1.0
    return _evaluate_eb_batch_numba(
        np.ascontiguousarray(r_arr, dtype=np.float64),
        float(t_s),
        np.ascontiguousarray(jitter_arr, dtype=np.float64),
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


def _dipole_basis_fields(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    k0: float,
    d_hat: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute regular Maxwell dipole modes Phi_dip and Psi_dip on (Nz, Ny, Nx)."""
    r = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
    kr = k0 * r
    j0 = spherical_jn(0, kr)
    j1 = spherical_jn(1, kr)
    j2 = spherical_jn(2, kr)

    nx = np.zeros_like(r)
    ny = np.zeros_like(r)
    nz = np.zeros_like(r)
    nonzero = r > 0.0
    np.divide(x_grid, r, out=nx, where=nonzero)
    np.divide(y_grid, r, out=ny, where=nonzero)
    np.divide(z_grid, r, out=nz, where=nonzero)

    dx, dy, dz = d_hat
    d_dot_n = dx * nx + dy * ny + dz * nz

    c_iso = j0 - 0.5 * j2
    c_rad = 1.5 * j2 * d_dot_n
    phi = np.empty((3, *r.shape), dtype=np.float64)
    phi[0] = c_iso * dx + c_rad * nx
    phi[1] = c_iso * dy + c_rad * ny
    phi[2] = c_iso * dz + c_rad * nz

    # d_hat x n_hat
    cross_x = dy * nz - dz * ny
    cross_y = dz * nx - dx * nz
    cross_z = dx * ny - dy * nx
    c_psi = 1.5 * j1
    psi = np.empty((3, *r.shape), dtype=np.float64)
    psi[0] = c_psi * cross_x
    psi[1] = c_psi * cross_y
    psi[2] = c_psi * cross_z
    return phi, psi


def _truncated_bidipole_kappa(f_number: float) -> float:
    rho = 1.0 / (2.0 * f_number)
    zeta = 0.5 * math.pi + math.atan(rho / 4.0 - 1.0 / rho)
    power_frac = (8.0 - (1.0 + math.cos(zeta)) ** 3) / 8.0
    return KAPPA_BIDIPOLE_MAX * math.sqrt(max(power_frac, 0.0))


def make_analytic_bidipole_pack(
    power_w: float = 1.0e14,
    lambda0_m: float = 0.8e-6,
    f_number: float = 0.25,
    pol: str = "z-linear",
    n_cycles_fwhm: float = 5.0,
    half_width_lambdas: float = 2.0,
    n_grid: int = 65,
    kappa: float | None = None,
    pulse_model: str = "gauss",
    t_ref_s: float = 0.0,
) -> FieldPack:
    """Generate an exact Maxwell-consistent bidipole FieldPack (LP or CP)."""
    if not math.isfinite(power_w) or power_w <= 0.0:
        raise ValueError(f"power_w must be finite and positive, got {power_w}")
    if not math.isfinite(lambda0_m) or lambda0_m <= 0.0:
        raise ValueError(f"lambda0_m must be finite and positive, got {lambda0_m}")
    if not math.isfinite(f_number) or f_number <= 0.0:
        raise ValueError(f"f_number must be finite and positive, got {f_number}")
    if pol not in ("z-linear", "circular"):
        raise ValueError(f"pol must be 'z-linear' or 'circular', got {pol!r}")
    if not math.isfinite(n_cycles_fwhm) or n_cycles_fwhm <= 0.0:
        raise ValueError(f"n_cycles_fwhm must be finite and positive, got {n_cycles_fwhm}")
    if not math.isfinite(half_width_lambdas) or half_width_lambdas <= 0.0:
        raise ValueError(f"half_width_lambdas must be finite and positive, got {half_width_lambdas}")
    if n_grid < 3:
        raise ValueError(f"n_grid must be >= 3, got {n_grid}")
    if pulse_model not in ("gauss", "none"):
        raise ValueError(f"pulse_model must be 'gauss' or 'none', got {pulse_model!r}")
    if not math.isfinite(t_ref_s):
        raise ValueError(f"t_ref_s must be finite, got {t_ref_s}")

    kappa_eff = _truncated_bidipole_kappa(f_number) if kappa is None else float(kappa)
    if not math.isfinite(kappa_eff) or kappa_eff <= 0.0:
        raise ValueError(f"kappa must be finite and positive, got {kappa_eff}")

    omega0 = omega_from_lambda(lambda0_m)
    k0 = omega0 / C_LIGHT
    t0_s = period_from_lambda(lambda0_m)
    tau_fwhm_s = n_cycles_fwhm * t0_s
    sigma_t_s = tau_fwhm_to_sigma_t(tau_fwhm_s)

    half_l = half_width_lambdas * lambda0_m
    step = 2.0 * half_l / (n_grid - 1)
    grid = GridSpec(
        x0_m=-half_l,
        dx_m=step,
        Nx=n_grid,
        y0_m=-half_l,
        dy_m=step,
        Ny=n_grid,
        z0_m=-half_l,
        dz_m=step,
        Nz=n_grid,
    )

    z_grid, y_grid, x_grid = np.meshgrid(grid.z_axis, grid.y_axis, grid.x_axis, indexing="ij")
    phi_z, psi_z = _dipole_basis_fields(x_grid, y_grid, z_grid, k0, (0.0, 0.0, 1.0))
    phi_x, psi_x = _dipole_basis_fields(x_grid, y_grid, z_grid, k0, (1.0, 0.0, 0.0))

    chi_max_lp_per_gev = theoretical_chi_max(1.0, power_w, lambda0_m, kappa=kappa_eff)
    gamma_1gev = energy_gev_to_gamma(1.0)
    e_max_lp = chi_max_lp_per_gev * E_CRIT / (2.0 * gamma_1gev)

    ec = np.zeros((1, 3, grid.Nz, grid.Ny, grid.Nx), dtype=np.complex128)
    bc = np.zeros((1, 3, grid.Nz, grid.Ny, grid.Nx), dtype=np.complex128)

    if pol == "z-linear":
        ec[0] = e_max_lp * (phi_z + 1.0j * psi_x)
        bc[0] = (e_max_lp / C_LIGHT) * (-phi_x + 1.0j * psi_z)
        chi_max_per_gev = chi_max_lp_per_gev
        a_max_expected = (
            A_BIDIPOLE_1PW_1UM
            * (kappa_eff / KAPPA_BIDIPOLE_MAX)
            * math.sqrt(power_w / 1.0e15)
            * (1.0e-6 / lambda0_m)
        )
    else:
        scale_cp = e_max_lp / math.sqrt(2.0)
        ec[0] = scale_cp * ((phi_z + 1.0j * psi_x) + 1.0j * (phi_x - 1.0j * psi_z))
        bc[0] = (1.0j / C_LIGHT) * ec[0]
        chi_max_per_gev = chi_max_lp_per_gev / math.sqrt(2.0)
        a_max_expected = (
            (A_BIDIPOLE_1PW_1UM / math.sqrt(2.0))
            * (kappa_eff / KAPPA_BIDIPOLE_MAX)
            * math.sqrt(power_w / 1.0e15)
            * (1.0e-6 / lambda0_m)
        )

    freqs = np.array([omega0], dtype=np.float64)
    e_ref = np.real(ec[0] * np.exp(1.0j * omega0 * t_ref_s))
    e_mag = np.sqrt(e_ref[0] ** 2 + e_ref[1] ** 2 + e_ref[2] ** 2)
    iz_max, iy_max, ix_max = np.unravel_index(np.argmax(e_mag), e_mag.shape)
    e_peak = float(e_mag[iz_max, iy_max, ix_max])
    a_max = e_field_to_a0(e_peak, omega0)
    max_loc = (
        float(grid.x_axis[ix_max]),
        float(grid.y_axis[iy_max]),
        float(grid.z_axis[iz_max]),
    )

    pulse_dict: dict[str, Any] = {
        "model": pulse_model,
        "convention": "intensity_fwhm",
        "t_ref_s": float(t_ref_s),
    }
    if pulse_model == "gauss":
        pulse_dict["tau_fwhm_s"] = float(tau_fwhm_s)
        pulse_dict["sigma_t_s"] = float(sigma_t_s)

    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "grid": {
            "x0_m": grid.x0_m,
            "dx_m": grid.dx_m,
            "Nx": grid.Nx,
            "y0_m": grid.y0_m,
            "dy_m": grid.dy_m,
            "Ny": grid.Ny,
            "z0_m": grid.z0_m,
            "dz_m": grid.dz_m,
            "Nz": grid.Nz,
        },
        "freqs": [float(omega0)],
        "power_W": float(power_w),
        "f_number": float(f_number),
        "lambda0_m": float(lambda0_m),
        "pol": pol,
        "pulse": pulse_dict,
        "checks": {
            "a_max": float(a_max),
            "a_max_expected": float(a_max_expected),
            "chi_max_per_GeV": float(chi_max_per_gev),
            "power_conservation_rel_err": 0.0,
            "max_location_m": list(max_loc),
        },
    }

    pulse = _parse_pulse_spec(pulse_dict)
    checks = FieldChecks(
        a_max=float(a_max),
        a_max_expected=float(a_max_expected),
        chi_max_per_GeV=float(chi_max_per_gev),
        power_conservation_rel_err=0.0,
        max_location_m=max_loc,
    )

    return FieldPack(
        Ec=np.ascontiguousarray(ec, dtype=np.complex128),
        Bc=np.ascontiguousarray(bc, dtype=np.complex128),
        grid=grid,
        freqs=freqs,
        power_W=float(power_w),
        f_number=float(f_number),
        lambda0_m=float(lambda0_m),
        pol=pol,
        pulse=pulse,
        checks=checks,
        metadata_raw=metadata,
    )
