"""Metadata construction and validation for optics field packs.

The metadata JSON is validated against ``data/schema.json`` (JSON Schema draft-07).
Because the project may run in environments where the third-party ``jsonschema``
package is unavailable, a small self-contained draft-07 validator is provided as the
default.  If ``jsonschema`` *is* importable it is used instead (it is the canonical
implementation listed in ``pixi.toml``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import constants as C

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schema.json"


def load_schema() -> dict:
    """Load and return ``data/schema.json`` as a dict."""
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate(instance: dict, schema: dict | None = None) -> None:
    """Validate ``instance`` against ``schema`` (default ``data/schema.json``).

    Raises ``ValueError`` with a descriptive message on failure.
    """
    if schema is None:
        schema = load_schema()
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(instance=instance, schema=schema)
    except ImportError:
        _validate_local(instance, schema)
    except Exception as exc:  # jsonschema raises its own ValidationError
        raise ValueError(f"metadata failed schema validation: {exc}") from exc


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema draft-07 validator (subset used by data/schema.json).
# --------------------------------------------------------------------------- #
def _validate_local(instance, schema, path: str = "$") -> None:
    if not isinstance(schema, dict):
        return

    if "const" in schema and instance != schema["const"]:
        raise ValueError(f"{path}: expected const {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        raise ValueError(f"{path}: {instance!r} not in enum {schema['enum']!r}")

    typ = schema.get("type")
    if typ is not None:
        _check_type(instance, typ, path)

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise ValueError(f"{path}: {instance} < minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise ValueError(f"{path}: {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise ValueError(f"{path}: {instance} > maximum {schema['maximum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise ValueError(f"{path}: array too short")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise ValueError(f"{path}: array too long")
        if "items" in schema and isinstance(schema["items"], dict):
            for i, item in enumerate(instance):
                _validate_local(item, schema["items"], f"{path}[{i}]")

    if isinstance(instance, dict):
        if "required" in schema:
            for key in schema["required"]:
                if key not in instance:
                    raise ValueError(f"{path}: missing required key {key!r}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in instance:
                _validate_local(instance[key], sub, f"{path}.{key}")


def _check_type(value, typ, path: str) -> None:
    ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(typ)
    if ok is None:
        return  # unsupported type keyword in this local subset: skip
    if not ok:
        raise ValueError(f"{path}: expected type {typ!r}, got {type(value).__name__}")


def build_metadata(
    *,
    lambda0_m: float,
    f_number: float,
    power_w: float,
    pol: str,
    grid: dict,
    checks: dict,
    pulse_model: str = "gauss",
    tau_fwhm_s: float | None = None,
    n_cycles: float | None = None,
    t_ref_s: float = 0.0,
    focal_length_m: float | None = None,
) -> dict:
    """Assemble the field-pack metadata dict conforming to ``data/schema.json``.

    The ``pulse`` block follows the intensity-FWHM convention of the paper: the field
    envelope sigma is ``sigma_t = tau_fwhm / sqrt(4 ln 2)``.
    """
    omega0 = C.omega_from_lambda(lambda0_m)
    if tau_fwhm_s is None and n_cycles is not None:
        tau_fwhm_s = C.tau_fwhm_from_cycles(n_cycles, lambda0_m)
    if tau_fwhm_s is None:
        raise ValueError("provide either tau_fwhm_s or n_cycles")

    pulse: dict = {
        "model": pulse_model,
        "convention": "intensity_fwhm",
        "tau_fwhm_s": float(tau_fwhm_s),
        "sigma_t_s": float(C.tau_fwhm_to_sigma_t(tau_fwhm_s)),
        "t_ref_s": float(t_ref_s),
    }

    meta: dict = {
        "schema_version": "1.0",
        "grid": grid,
        "freqs": [float(omega0)],
        "power_W": float(power_w),
        "f_number": float(f_number),
        "lambda0_m": float(lambda0_m),
        "pol": pol,
        "pulse": pulse,
        "checks": checks,
        # Extra (non-schema) reporting fields shown in the outs.md sketch.
        "convention": "Re[E * exp(i*omega*t)]",
        "theta_max_rad": float(2.0 * math.atan(1.0 / (4.0 * f_number))),
    }
    if n_cycles is not None:
        meta["n_cycles"] = float(n_cycles)
    if focal_length_m is not None:
        meta["R_max_m"] = float(focal_length_m / (2.0 * f_number))

    validate(meta)
    return meta

