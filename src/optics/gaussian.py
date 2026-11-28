"""Gaussian laser source model and incident-beam amplitude profiles.

The *realistic* optical pipeline models the laser as a collimated Gaussian beam that
is spatially shaped by an apodizer *before* reflecting off the parabolic mirror.  The
analytic Sheppard-Larkin profile ``I^p(R) ~ ((R/2L)^2 + 1)^-4`` (eq. 13 of Olofsson &
Gonoskov, Phys. Rev. A 106, 063512, 2022) is kept here as the *ideal* reference
baseline used for the quantitative comparison.

All radial profiles use the normalized mirror radius ``r = R / R_max``, so they are
independent of the mirror focal length.
"""

from __future__ import annotations

import numpy as np


def gaussian_source(R, w0):
    """Collimated Gaussian field amplitude exp(-(R/w0)^2) (1/e field radius w0)."""
    return np.exp(-((R / w0) ** 2))


def apodizer(R, R_max, order=1.0):
    """Apodizer transmission exp(-(R/R_max)^(2*order)).

    A super-Gaussian of order ``order``; a large order approaches a hard aperture.
    """
    if order <= 0.0:
        raise ValueError("order must be positive")
    return np.exp(-((R / R_max) ** (2.0 * order)))


def gaussian_incident_amplitude(r_over_rmax, *, w0_over_rmax, apod_order):
    """Realistic incident field amplitude (Gaussian x apodizer) on the mirror.

    ``r_over_rmax`` is the normalized radius R / R_max.
    """
    return gaussian_source(r_over_rmax, w0_over_rmax) * apodizer(
        r_over_rmax, 1.0, order=apod_order
    )


def ideal_incident_amplitude(r_over_rmax, *, f_number):
    """Ideal Sheppard-Larkin incident field amplitude sqrt(I^p(R)) (eq. 13).

    ``I^p(R) ~ ((R/2L)^2 + 1)^-4`` with ``R/2L = (R/R_max) / (4 f-number)``.
    """
    r_over_2L = r_over_rmax / (4.0 * f_number)
    return (r_over_2L ** 2 + 1.0) ** -2