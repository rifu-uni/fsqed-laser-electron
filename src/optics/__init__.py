"""Optics module: bi-dipole focusing simulation (Olofsson & Gonoskov, PRA 106, 063512).

Public API
----------
``constants``   : SI constants and QED scaling relations.
``focus``       : bi-dipole far field, parabolic-mirror geometry, vectorial focal-field
                  integral (Debye), ``B = (i/omega) curl E``, checks.
``gaussian``    : Gaussian source + paraxial shaping pipeline (thin lens, apodizer, FFT).
``sweep``       : ``kappa_eff(f-number)`` sweep for the Fig. 4 comparison.
``metadata``    : metadata construction + validation against ``data/schema.json``.
``cli``         : field-pack generation and JSON output.
"""

from __future__ import annotations

from . import constants, focus, gaussian, metadata, sweep
from .cli import default_parameters, generate_field_pack, main, save_field_pack

__all__ = [
    "constants",
    "focus",
    "gaussian",
    "sweep",
    "metadata",
    "default_parameters",
    "generate_field_pack",
    "save_field_pack",
    "main",
]

