"""Optics module: bi-dipole focusing simulation (Olofsson & Gonoskov, PRA 106, 063512).

Public API
----------
constants   : SI constants and QED scaling relations.
focus       : vectorial focal-field (Debye) integral; far-field models (analytic
              Sheppard-Larkin and realistic Gaussian + apodizer + mirror pipeline).
gaussian    : Gaussian laser source and incident-beam amplitude profiles.
sweep       : kappa_eff(f-number) sweep for the Fig. 4 comparison (both modes).
comparison  : MSE / correlation / transmission between realistic and ideal profiles.
metadata    : metadata construction + validation against data/schema.json.
cli         : field-pack generation and JSON output (--mode ideal|gaussian).
"""

from __future__ import annotations

from . import comparison, constants, focus, gaussian, metadata, sweep
from .cli import default_parameters, generate_field_pack, main, save_field_pack

__all__ = [
    "constants",
    "focus",
    "gaussian",
    "sweep",
    "comparison",
    "metadata",
    "default_parameters",
    "generate_field_pack",
    "save_field_pack",
    "main",
]