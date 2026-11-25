"""Vectorial focal-field computation for the "bi-dipole" wave.

Implements the optics of Olofsson & Gonoskov, Phys. Rev. A 106, 063512 (2022),
Secs. III-IV:

* The far-field angular distribution of the bi-dipole wave (eqs. 9-10), i.e. the
  analytic Sheppard-Larkin input.
* The parabolic-mirror geometry (normal ``N = (y_hat - n)/|y_hat - n|``) and the
  local-normal polarization reflection ``E^p = E - 2 (E . N) N``.
* The vectorial focal-field integral (Debye / Richards-Wolf), evaluated exactly
  (non-paraxial) through the angular-spectrum method: the monochromatic phasor
  ``Ec(x, y, z)`` is a superposition of plane waves with the exact propagator
  ``ky = sqrt(k^2 - kx^2 - kz^2)`` (never the paraxial ``k - (kx^2+kz^2)/2k``).

The magnetic phasor ``Bc`` is reconstructed from ``Ec`` via ``B = (i/omega) curl E``
using central finite differences, matching the convention used by ``src.particles``.

Global conventions (see README):

* Electron travels along ``+y``; laser is incident along ``-y`` and converges to the
  focus at the origin; polarization along ``z``.
* Phasor convention ``E = Re[Ec * exp(+i omega t)]``.
* Field arrays have shape ``(Nf, 3, Nz, Ny, Nx)``; axis 1 is ``(x, y, z)``; grid axes
  are slow-to-fast ``(z, y, x)``.
"""

from __future__ import annotations

import numpy as np

from . import constants as C


# --------------------------------------------------------------------------- #
# Analytic bi-dipole far field (Sheppard-Larkin input), eqs. (9)-(10).
# --------------------------------------------------------------------------- #
def bi_dipole_far_field(n: np.ndarray) -> np.ndarray:
    """Far-field electric phasor of the bi-dipole wave.

    Parameters
    ----------
    n : np.ndarray, shape (..., 3)
        Unit vector(s) pointing *from the focus toward the source*; the incoming
        radiation arrives from the ``n_y < 0`` hemisphere. Components ``(nx, ny, nz)``.

    Returns
    -------
    np.ndarray, shape (..., 3)
        Complex phasor ``(Ex, Ey, Ez)`` (dimensionless shape; the absolute amplitude
        is fixed later by power normalization).
    """
    nx = n[..., 0]
    ny = n[..., 1]
    nz = n[..., 2]
    # E^e = (z_hat x n) x n  (electric-dipole wave);  E^b = x_hat x n (magnetic-dipole wave).
    ex = nx * nz
    ey = nz * (ny - 1.0)
    ez = ny - nx * nx - ny * ny  # == ny - 1 + nz^2
    return np.stack([ex, ey, ez], axis=-1)


def f_number_to_theta_max(f_number: float) -> float:
    """Parabolic-mirror semi-aperture angle ``theta_max`` (rad) from the f-number.

    The paper defines ``f-number = L / (2 R_max)`` (``L`` focal length, ``R_max``
    mirror radius).  A paraxial-ray at radius ``R`` reflects at angle ``theta`` from the
    axis with ``R = 2 L tan(theta/2)``, hence
    ``theta_max = 2 atan(R_max/(2 L)) = 2 atan(1/(4 f-number))``.
    """
    if not np.isfinite(f_number) or f_number <= 0.0:
        raise ValueError(f"f_number must be finite and positive, got {f_number}")
    return 2.0 * np.arctan(1.0 / (4.0 * f_number))


def mirror_normal(n: np.ndarray) -> np.ndarray:
    """Local unit normal of the parabolic mirror, ``N = (y_hat - n)/|y_hat - n|``."""
    y_hat = np.zeros_like(n)
    y_hat[..., 1] = 1.0
    v = y_hat - n
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norm == 0.0, 1.0, norm)


def reflect(E: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Mirror reflection with local-normal polarization rotation ``E^p = E - 2 (E . N) N``."""
    dot = np.sum(E * N, axis=-1, keepdims=True)
    return E - 2.0 * dot * N


# --------------------------------------------------------------------------- #
# Focal-field integral (Debye / angular spectrum).
# --------------------------------------------------------------------------- #
def _centered_axis(span_m: float, npts: int) -> tuple[float, float]:
    """Return ``(start, step)`` for a centered axis with ``npts`` points spanning ``span_m``."""
    step = span_m / npts
    start = -(npts // 2) * step
    return start, step


def focal_field_angular_spectrum(
    *,
    lambda0_m: float,
    f_number: float,
    power_w: float,
    nx: int = 128,
    ny: int = 128,
    nz: int = 128,
    span_xy_m: float | None = None,
    span_y_m: float | None = None,
    ntheta: int = 40,
    nphi: int = 80,
    far_field=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute the monochromatic focal phasors ``Ec`` and ``Bc`` of a focused beam.

    The field is the exact (non-paraxial) Debye integral of the far-field pattern
    ``far_field(n)`` over the parabolic-mirror aperture set by ``f_number``, evaluated
    on a Cartesian grid by direct ``(theta, phi)`` quadrature of the angular-spectrum
    (plane-wave) integral.  The result is normalized so that the time-averaged power
    flowing through the focal plane (``y = 0``) equals ``power_w``.

    Parameters
    ----------
    lambda0_m : float
        Carrier wavelength (m).
    f_number : float
        Mirror f-number ``L/(2 R_max)``; ``0.25`` collects the full ``-y`` hemisphere.
    power_w : float
        Power delivered to the focus (W), fixing the absolute amplitude.
    nx, ny, nz : int
        Grid sizes (must be even for the centered-axis convention).
    span_xy_m : float, optional
        Transverse ``x``/``z`` span (m).  Default ``8 * lambda0_m``.
    span_y_m : float, optional
        Longitudinal ``y`` span (m).  Default ``8 * lambda0_m``.
    far_field : callable, optional
        ``far_field(n) -> E(..., 3)``.  Defaults to :func:`bi_dipole_far_field`.

    Returns
    -------
    Ec : np.ndarray, shape (1, 3, nz, ny, nx), complex128
        Electric field phasor (V/m).
    Bc : np.ndarray, shape (1, 3, nz, ny, nx), complex128
        Magnetic field phasor (T).
    grid : dict
        ``{"x0_m", "dx_m", "Nx", ...}`` describing the Cartesian grid.
    """
    if far_field is None:
        far_field = bi_dipole_far_field

    if any(int(v) < 2 for v in (nx, ny, nz)):
        raise ValueError("grid sizes must be >= 2")
    if any(int(v) % 2 != 0 for v in (nx, ny, nz)):
        raise ValueError("grid sizes nx, ny, nz must be even for the centered-axis convention")

    k = 2.0 * np.pi / lambda0_m
    omega = C.omega_from_lambda(lambda0_m)
    theta_max = f_number_to_theta_max(f_number)

    if span_xy_m is None:
        span_xy_m = 8.0 * lambda0_m
    if span_y_m is None:
        span_y_m = 8.0 * lambda0_m

    x0, dx = _centered_axis(span_xy_m, nx)
    y0, dy = _centered_axis(span_y_m, ny)
    z0, dz = _centered_axis(span_xy_m, nz)
    X = x0 + np.arange(nx) * dx
    Y = y0 + np.arange(ny) * dy
    Z = z0 + np.arange(nz) * dz

    # Ray grid: theta measured from the -y axis (the mirror axis), phi azimuth.  This
    # (theta, phi) parameterization makes the Debye integral a plain, singularity-free
    # sum of plane waves (no 1/ky Jacobian pole at the NA=1 rim).
    th_edges = np.linspace(0.0, theta_max, ntheta + 1)
    th = 0.5 * (th_edges[:-1] + th_edges[1:])
    ph = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack(
        [np.sin(TH) * np.cos(PH), -np.cos(TH), np.sin(TH) * np.sin(PH)], axis=-1
    )
    far = far_field(n)  # (ntheta, nphi, 3)
    d_theta = th_edges[1] - th_edges[0]
    w = np.sin(TH) * d_theta * (2.0 * np.pi / nphi)  # solid-angle quadrature weight

    # Power normalization: P = (1/(2 eta0)) int |E_inf|^2 dOmega, with E_inf = C * far.
    I2 = float(np.sum(np.sum(np.abs(far) ** 2, axis=-1) * w))
    if I2 <= 0.0:
        raise ValueError("far-field pattern carries zero power over the aperture")
    Cscale = np.sqrt(2.0 * C.ETA_0 * power_w / I2)  # far-field amplitude scale (V)

    # Debye integral: E(r) = -(i/lambda) C sum_{theta,phi} w far(n) e^{-ik n.r}.
    pref = -1j * Cscale / lambda0_m
    amp = pref * w[..., None] * far  # (ntheta, nphi, 3)

    Ec = np.zeros((3, nz, ny, nx), dtype=np.complex128)
    for i in range(ntheta):
        sth = np.sin(th[i])
        # Phase factors that depend only on phi (shared by the theta ring).
        gx = np.exp(-1j * k * np.outer(sth * np.cos(ph), X))  # (nphi, nx)
        gz = np.exp(-1j * k * np.outer(sth * np.sin(ph), Z))  # (nphi, nz)
        ph_y = np.exp(-1j * k * (-np.cos(th[i])) * Y)  # (ny,)
        for j in range(nphi):
            phase = gz[j][:, None, None] * ph_y[None, :, None] * gx[j][None, None, :]
            for comp in range(3):
                Ec[comp] += amp[i, j, comp] * phase

    Bc = magnetic_field_from_curl(Ec, dx, dy, dz, omega)

    grid = {
        "x0_m": float(x0), "dx_m": float(dx), "Nx": int(nx),
        "y0_m": float(y0), "dy_m": float(dy), "Ny": int(ny),
        "z0_m": float(z0), "dz_m": float(dz), "Nz": int(nz),
    }
    return Ec[None, ...], Bc[None, ...], grid


def focal_field_slice(
    *,
    lambda0_m: float,
    f_number: float,
    power_w: float,
    plane: str = "y",
    coord_m: float = 0.0,
    n1: int = 256,
    n2: int = 256,
    span1_m: float | None = None,
    span2_m: float | None = None,
    ntheta: int = 64,
    nphi: int = 128,
    far_field=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute the electric phasor ``Ec`` on a 2-D slice through the focus.

    This is the same Debye integral as :func:`focal_field_angular_spectrum` but
    evaluated only on one plane, so it is much cheaper and allows high in-plane
    resolution for visualisation.

    Parameters
    ----------
    plane : str
        ``"y"`` -> ``x-z`` plane at ``y = coord_m`` (axis1 = x, axis2 = z);
        ``"x"`` -> ``y-z`` plane at ``x = coord_m`` (axis1 = z, axis2 = y);
        ``"z"`` -> ``x-y`` plane at ``z = coord_m`` (axis1 = x, axis2 = y).
    coord_m : float
        Fixed coordinate of the slice (m).  Default 0 (through the focus).

    Returns
    -------
    Ec : np.ndarray, shape (3, n2, n1), complex128
        Field phasor ``(Ex, Ey, Ez)`` on the slice.
    axis1 : np.ndarray, shape (n1,)
        First transverse axis coordinates (m).
    axis2 : np.ndarray, shape (n2,)
        Second axis coordinates (m).
    info : dict
        ``{"plane", "coord_m", "grid"}`` (grid maps axis names to ``(start, step, n)``).
    """
    if far_field is None:
        far_field = bi_dipole_far_field

    k = 2.0 * np.pi / lambda0_m
    theta_max = f_number_to_theta_max(f_number)

    if span1_m is None:
        span1_m = 8.0 * lambda0_m
    if span2_m is None:
        span2_m = 8.0 * lambda0_m

    a1_start, a1_step = _centered_axis(span1_m, n1)
    a2_start, a2_step = _centered_axis(span2_m, n2)
    axis1 = a1_start + np.arange(n1) * a1_step
    axis2 = a2_start + np.arange(n2) * a2_step

    # Map plane -> axis names; fixed coordinate is the remaining axis.
    plane_map = {
        "y": ("x", "z", "y"),
        "x": ("z", "y", "x"),
        "z": ("x", "y", "z"),
    }
    if plane not in plane_map:
        raise ValueError(f"plane must be one of {sorted(plane_map)}, got {plane!r}")
    a1_name, a2_name, fixed_name = plane_map[plane]

    # Ray grid (theta from the -y axis, phi azimuth).
    th_edges = np.linspace(0.0, theta_max, ntheta + 1)
    th = 0.5 * (th_edges[:-1] + th_edges[1:])
    ph = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(TH) * np.cos(PH), -np.cos(TH), np.sin(TH) * np.sin(PH)], axis=-1)
    far = far_field(n)  # (ntheta, nphi, 3)
    w = np.sin(TH) * (th_edges[1] - th_edges[0]) * (2.0 * np.pi / nphi)

    I2 = float(np.sum(np.sum(np.abs(far) ** 2, axis=-1) * w))
    if I2 <= 0.0:
        raise ValueError("far-field pattern carries zero power over the aperture")
    Cscale = np.sqrt(2.0 * C.ETA_0 * power_w / I2)
    pref = -1j * Cscale / lambda0_m
    amp = pref * w[..., None] * far  # (ntheta, nphi, 3)

    Ec = np.zeros((3, n2, n1), dtype=np.complex128)
    for i in range(ntheta):
        sth = np.sin(th[i])
        cth = np.cos(th[i])
        for j in range(nphi):
            nxv = sth * np.cos(ph[j])
            nyv = -cth
            nzv = sth * np.sin(ph[j])
            n_comp = {"x": nxv, "y": nyv, "z": nzv}
            # 2-D phase = exp(-i k (n_a1 a1 + n_a2 a2 + n_fixed coord_m)).
            f1 = np.exp(-1j * k * n_comp[a1_name] * axis1)   # (n1,)
            f2 = np.exp(-1j * k * n_comp[a2_name] * axis2)   # (n2,)
            fixed = np.exp(-1j * k * n_comp[fixed_name] * coord_m)
            phase = f2[:, None] * f1[None, :] * fixed        # (n2, n1)
            for comp in range(3):
                Ec[comp] += amp[i, j, comp] * phase

    grid = {
        a1_name: (float(a1_start), float(a1_step), int(n1)),
        a2_name: (float(a2_start), float(a2_step), int(n2)),
        fixed_name: (float(coord_m), 0.0, 1),
    }
    info = {"plane": plane, "coord_m": float(coord_m), "grid": grid}
    return Ec, axis1, axis2, info


def magnetic_field_from_curl(
    Ec: np.ndarray, dx: float, dy: float, dz: float, omega: float
) -> np.ndarray:
    """Compute ``Bc = (i/omega) curl(Ec)`` with central finite differences.

    ``Ec`` has shape ``(3, nz, ny, nx)`` with component axis ``(x, y, z)`` and grid
    axes ``(z, y, x)``.
    """
    Ex = Ec[0]
    Ey = Ec[1]
    Ez = Ec[2]

    dEz_dy = np.gradient(Ez, dy, axis=1)
    dEy_dz = np.gradient(Ey, dz, axis=0)
    dEx_dz = np.gradient(Ex, dz, axis=0)
    dEz_dx = np.gradient(Ez, dx, axis=2)
    dEy_dx = np.gradient(Ey, dx, axis=2)
    dEx_dy = np.gradient(Ex, dy, axis=1)

    fac = 1j / omega
    Bc = np.zeros_like(Ec)
    Bc[0] = fac * (dEz_dy - dEy_dz)  # Bx
    Bc[1] = fac * (dEx_dz - dEz_dx)  # By
    Bc[2] = fac * (dEy_dx - dEx_dy)  # Bz
    return Bc


def field_power_through_y_plane(
    Ec: np.ndarray, Bc: np.ndarray, dx: float, dz: float, iy: int | None = None
) -> float:
    """Time-averaged power (W) flowing through a ``y = const`` plane.

    ``S_y = Re(Ez Bx* - Ex Bz*) / (2 mu0)``.
    """
    if iy is None:
        iy = Ec.shape[2] // 2
    s_y = np.real(
        Ec[2, :, iy, :] * np.conj(Bc[0, :, iy, :])
        - Ec[0, :, iy, :] * np.conj(Bc[2, :, iy, :])
    ) / (2.0 * C.MU_0)
    return float(np.sum(s_y) * dx * dz)


def compute_checks(
    Ec: np.ndarray,
    *,
    lambda0_m: float,
    power_w: float,
    grid: dict,
    Bc: np.ndarray | None = None,
) -> dict:
    """Derive the self-validation metadata from a computed field pack.

    Returns a ``checks`` dict with ``a_max``, ``a_max_expected``,
    ``chi_max_per_GeV``, ``power_conservation_rel_err`` and ``max_location_m``.
    """
    omega = C.omega_from_lambda(lambda0_m)
    E_rel = C.relativistic_field_scale(omega)

    E = Ec[0]  # strip Nf axis -> (3, nz, ny, nx)
    B = None if Bc is None else Bc[0]

    emag = np.abs(E)
    idx = np.unravel_index(int(np.argmax(emag)), emag.shape)
    comp, iz, iy, ix = (int(v) for v in idx)
    a_max = float(emag[comp, iz, iy, ix] / E_rel)

    # chi ~ 2 gamma |E_perp| / E_cr  (head-on collision, electron along +y).
    e_perp = np.sqrt(np.abs(E[0]) ** 2 + np.abs(E[2]) ** 2)
    chi_max_per_gev = 2.0 * C.GAMMA_PER_GEV * float(np.max(e_perp)) / C.E_CRIT

    # Peak location in meters (centered-axis convention).
    max_loc = [
        float(grid["x0_m"] + ix * grid["dx_m"]),
        float(grid["y0_m"] + iy * grid["dy_m"]),
        float(grid["z0_m"] + iz * grid["dz_m"]),
    ]

    # Power conservation across two different y planes.
    if B is not None:
        nyy = E.shape[2]  # y axis of (3, nz, ny, nx)
        p0 = field_power_through_y_plane(E, B, grid["dx_m"], grid["dz_m"], iy=nyy // 2)
        p1 = field_power_through_y_plane(E, B, grid["dx_m"], grid["dz_m"], iy=max(nyy // 4, 0))
        p_err = float(abs(p0 - p1) / p0) if p0 > 0.0 else 0.0
    else:
        p_err = 0.0

    # a0 is lambda-independent for a fixed f-number: a0 = eE/(mc omega) ~ sqrt(P).
    a_max_expected = float(C.A_BIDIPOLE_1PW_1UM * np.sqrt(power_w / 1.0e15))

    return {
        "a_max": a_max,
        "a_max_expected": a_max_expected,
        "chi_max_per_GeV": chi_max_per_gev,
        "power_conservation_rel_err": p_err,
        "max_location_m": max_loc,
    }


