# fsqed-laser-electron

Numerical reproduction of Olofsson & Gonoskov, *Attaining a strong-field QED signal at laser-electron colliders with optimized focusing*, Phys. Rev. A **106**, 063512 (2022).

## Modules

`src/optics/`

1. Gaussian source model (`gaussian.py`): a collimated Gaussian beam, configurable for
   FSQED lasers (tens-of-fs pulse duration, PW-class peak power).
2. Realistic optical pipeline (`focus.py`): the Gaussian beam is shaped by an apodizer
   and reflected off the parabolic mirror via the local-normal polarization rotation
   `E^p = E - 2 (E . N) N` (mirror normal `N = (y_hat - n)/|y_hat - n|`), then
   propagated through the vectorial Debye focal-field integral (non-paraxial) to obtain
   the complex phasor `Ec(x, y, z)`.
3. Ideal baseline: the analytic Sheppard-Larkin far field (eqs. 9-10, equivalent to
   reflecting the intensity profile `I^p(R) ~ ((R/2L)^2 + 1)^-4` of eq. 13) is kept as
   the theoretical reference for comparison.
4. `kappa_eff(f-number)` sweep (Fig. 4) for *both* models.
5. Quantitative comparison (`comparison.py`): mean squared error (MSE), Pearson
   correlation and apodizer power transmission between the realistic and ideal
   incident-beam profiles.

`src/particles/`

1. Electron trajectory pusher (Boris/Vay), impact parameter sampled from `p(b) = 2b/b_max^2`, timing jitter `delta t` per shot.
2. `chi(b,t)` traces and the time fraction `F = (1/T) int Theta[chi - chi_max/2] dt`.
3. QED extension: stochastic photon emission from `W_rad(gamma, chi)`, photon energy and deviation angle `alpha ~ a_max/gamma`, selection cuts, `sigma = N_signal/n_e tau_l c` and `eta = N_signal/N_total`.
4. Shot-count / `3 sigma` statistics; benchmark digitization (Figs. 4-5).

## Execution modes

The optics module exposes two execution modes through the CLI
(`python -m optics`, or `python src/optics/cli.py`):

```bash
# Realistic optical pipeline (default): Gaussian beam + apodizer + mirror reflection
python -m optics --mode gaussian --power-w 1e15 --tau-fwhm-s 30e-15 --w0-over-rmax 0.7 --apod-order 8

# Ideal analytic baseline: Sheppard-Larkin far field injected directly
python -m optics --mode ideal
```

For the Gaussian mode, the output metadata includes a `degradation` block with the MSE,
the Pearson correlation and the apodizer transmission against the ideal Sheppard-Larkin
profile.  The `kappa_eff(f-number)` sweep can be run for both modes with
`optics.sweep.kappa_eff_vs_f_number(..., far_field_builder=...)`.

### Schema

The particles module receives field packs. A pack is one `.npz` file per f-number
(`data/f{fnumber}.npz`, gitignored) containing `Ec` (required), `Bc` (optional), and
`metadata` (a JSON string validated by `data/schema.json`):

- `Ec`: `(Nf, 3, Nz, Ny, Nx)` `complex128` electric field phasor in V/m (vector
  components `(Ex, Ey, Ez)` on axis 1). Legacy 4D scalar `Ez` packs are auto-promoted
  by the loader when `pol == "z-linear"`.
- `Bc` (optional): `(Nf, 3, Nz, Ny, Nx)` `complex128` magnetic field phasor in T. If
  omitted, `src/particles` reconstructs `Bc = (i/omega) curl Ec` on load.

Time is never passed as data, but reconstructed using

    E(x,y,z,t) = g(t) sum_k Re[ Ec^(k)(x,y,z) exp(+i omega_k t) ]

at the pusher's own timestep (`dt <= T0/20`, `~0.13 fs` at 800 nm). Timing jitter is a
per-shot phase/time shift `delta t`.

### Metadata

Save the data with the validated schema:

```python
np.savez_compressed("fields/f0.25.npz",
    Ec=Ec,                                   # (Nf, 3, Nz, Ny, Nx) complex128, V/m
    Bc=Bc,                                   # optional (Nf, 3, Nz, Ny, Nx) complex128, T
    metadata=json.dumps(meta))               # meta conforms to data/schema.json
```

Also validated at load time.

## Development and execution

Package management is done via `pixi`. Each module is independent from each other, then
imported into centralized execution notebooks.

Run the tests with:

```bash
python -m pytest -q
```

## Conventions (global)

- SI units in all data; relativistic amplitudes (`a0 = eE/mc/omega0`) only as derived
  reporting quantities.
- Phasor convention: `E = Re[Ec * exp(+i omega t)]`.
- Grid axes slow-to-fast: `(Nf, Nz, Ny, Nx)`. Electron along `+y`; laser incident along
  `-y`; polarization along `z`.
- Pulse durations quoted as **intensity FWHM**; the field-envelope sigma is
  `sigma_t = tau_fwhm / sqrt(4 ln 2) ~ tau_fwhm / 1.665`.