# fsqed-laser-electron

Numerical reproduction of Olofsson & Gonoskov, *Attaining a strong-field QED signal at laser-electron colliders with optimized focusing*, Phys. Rev. A **106**, 063512 (2022).

## Modules

`src/optics/`

1. Gaussian source model (`gaussian.py`): a collimated Gaussian beam, configurable for
   FSQED lasers (tens-of-fs pulse duration, PW-class peak power).
2. Realistic optical pipeline (`focus.py`): the Gaussian beam is shaped by an apodizer
   and reflected off the parabolic mirror via the local-normal polarization rotation
   $\mathbf{E}^p = \mathbf{E} - 2(\mathbf{E}\cdot\mathbf{N})\mathbf{N}$ (mirror normal
   $\mathbf{N} = (\hat{\mathbf{y}} - \mathbf{n})/|\hat{\mathbf{y}} - \mathbf{n}|$), then
   propagated through the vectorial Debye focal-field integral (non-paraxial) to obtain
   the complex phasor $\mathbf{E}_c(x,y,z)$ (and exact $\mathbf{B}_c(x,y,z)$).
3. Ideal baseline: the analytic Sheppard–Larkin far field (eqs. 9–10, equivalent to
   reflecting the intensity profile $I^p(R) \propto ((R/2L)^2 + 1)^{-4}$ of eq. 13) is
   kept as the theoretical reference for comparison.
4. $\kappa_{\mathrm{eff}}(f\text{-number})$ sweep (Fig. 4) for *both* models.
5. Quantitative comparison (`comparison.py`): mean squared error (MSE), Pearson
   correlation, and apodizer power transmission between the realistic and ideal
   incident-beam profiles.

`src/particles/`

1. Electron trajectory pusher (Boris/Vay), impact parameter sampled from $p(b) = 2b/b_{\max}^2$, timing jitter $\delta t$ per shot.
2. $\chi(b,t)$ traces and the time fraction $F = \tfrac{1}{T}\int\Theta\!\left[\chi - \chi_{\max}/2\right]dt$.
3. QED extension: stochastic photon emission from $W_{\mathrm{rad}}(\gamma,\chi)$, photon energy and deviation angle $\alpha \sim a_{\max}/\gamma$, selection cuts, $\sigma = N_{\mathrm{signal}}/n_e\tau_l c$ and $\eta = N_{\mathrm{signal}}/N_{\mathrm{total}}$.
4. Shot-count / $3\sigma$ statistics; benchmark digitization (Figs. 4–5).

## Execution modes

The optics module exposes two execution modes through the CLI
(`pixi run python -m src.optics`):

```bash
# Realistic optical pipeline (default): Gaussian beam + apodizer + mirror reflection
pixi run python -m src.optics --mode gaussian --power-w 1e15 --n-cycles 5 --w0-over-rmax 0.7 --apod-order 8

# Ideal analytic baseline: Sheppard-Larkin far field injected directly
pixi run python -m src.optics --mode ideal
```

For the Gaussian mode, the output metadata includes a `degradation` block with the MSE,
the Pearson correlation, and the apodizer transmission against the ideal Sheppard–Larkin
profile. The $\kappa_{\mathrm{eff}}(f\text{-number})$ sweep can be run for both modes with
`src.optics.sweep.kappa_eff_vs_f_number(..., far_field_builder=...)`.

### Schema

The particles module receives field packs. A pack is one `.npz` file per $f$-number
(`data/f{fnumber}.npz`, gitignored) containing `Ec` (required), `Bc` (optional), and
`metadata` (a JSON string validated by `data/schema.json`):

- `Ec`: `(Nf, 3, Nz, Ny, Nx)` `complex128` electric field phasor in $\mathrm{V/m}$ (vector
  components `(Ex, Ey, Ez)` on axis `1`). Legacy 4D `(Nf, Nz, Ny, Nx)` scalar $E_z$
  packs are auto-promoted by the loader when `pol == "z-linear"`.
- `Bc` (optional): `(Nf, 3, Nz, Ny, Nx)` `complex128` magnetic field phasor in $\mathrm{T}$.
  Included by `src/optics` via exact Debye quadrature
  $\mathbf{B}_{\mathrm{far}}(\mathbf{n}) = +\frac{1}{c}\,\mathbf{n}\times\mathbf{E}_{\mathrm{far}}(\mathbf{n})$;
  if omitted in custom `.npz` packs, `src/particles` reconstructs
  $\mathbf{B}_c^{(k)} = \frac{i}{\omega_k}\nabla \times \mathbf{E}_c^{(k)}$ on load
  via central finite differences.

Time is never passed as data, but reconstructed using

$$\mathbf{E}(x,y,z,t) = g(t) \sum_k \mathrm{Re}\!\left[\mathbf{E}_c^{(k)}(x,y,z)\, e^{+i\omega_k t}\right]$$

at the pusher's own timestep ($\Delta t \le T_0/20$, $\lesssim 0.13\,\mathrm{fs}$ at 800 nm).
Timing jitter is a per-shot phase/time shift $\delta t$.

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
pixi run test
```

## Conventions (global)

- SI units in all data; relativistic amplitudes ($a_0 = eE/mc\omega_0$) only as derived
  reporting quantities.
- Phasor convention: $E = \mathrm{Re}[E_c\, e^{+i\omega t}]$.
- Grid axes slow-to-fast: `(Nf, Nz, Ny, Nx)`. Electron along $+\hat{y}$; laser incident
  along $-\hat{y}$; polarization along $\hat{z}$.
- Pulse durations quoted as **intensity FWHM** (the paper's convention); the
  field-envelope sigma is $\sigma_t = \tau_{\mathrm{FWHM}}/\sqrt{4\ln 2} \approx \tau_{\mathrm{FWHM}}/1.665$.
  The schema pins this.
