# fsqed-laser-electron

Numerical reproduction of Olofsson & Gonoskov, *Attaining a strong-field QED signal at laser-electron colliders with optimized focusing*, Phys. Rev. A **106**, 063512 (2022).

## Modules

>[!warning]
> clanker draft

`src/optics/`

1. Gaussian source model.
2. Mirror reflection + vectorial focal-field integral: the complex phasor $E_c(x,y,z)$ at $\omega_0$, including the local-normal polarization rotation $\mathbf{E}^p = \mathbf{E} - 2(\mathbf{E}\cdot\mathbf{N})\mathbf{N}$. No scalar-paraxial code past the parabola — FFT-style propagation shapes the beam *before* the mirror only.
3. $\kappa_{\mathrm{eff}}(f\#)$ sweep → comparison with Fig. 4. Headline plot; starts from the analytic Sheppard–Larkin input.
4. Paraxial shaping pipeline: thin lens $t(r) = e^{-ikr^2/2f}$, apodizer, propagation $U' = \mathcal{F}^{-1}\{\mathcal{F}\{U\}\,e^{-i\Delta z(k_x^2+k_y^2)/2k}\}$.
5. Apodizer transmission / power accounting.

`src/particles/`

1. Electron trajectory pusher (Boris/Vay), impact parameter sampled from $p(b) = 2b/b_{\max}^2$, timing jitter $\delta t$ per shot.
2. $\chi(b,t)$ traces and the time fraction $F = \tfrac{1}{T}\int\Theta\!\left[\chi - \chi_{\max}/2\right]dt$.
3. QED extension: stochastic photon emission from $W_{\mathrm{rad}}(\gamma,\chi)$, photon energy and deviation angle $\alpha \sim a_{\max}/\gamma$, selection cuts, $\sigma = N_{\mathrm{signal}}/n_e\tau_l c$ and $\eta = N_{\mathrm{signal}}/N_{\mathrm{total}}$.
4. Shot-count / $3\sigma$ statistics; benchmark digitization (Figs. 4–5).

### Schema

The particles module receives field packs. A pack is one `.npz` file per $f$-number (`data/f{fnumber}.npz`, gitignored) containing two keys: `Ec` (the array) and `metadata` (a JSON string validated by `data/schema.json`).

>[!warning]
> check this portion, also clanker made

Time is never passed as data, but reconstructed using

$$E(x,y,z,t) = \sum_k \mathrm{Re}\!\left[E_c^{(k)}(x,y,z)\, e^{+i\omega_k t}\right]$$

at the pusher's own timestep ($\Delta t \ll$ quarter-carrier period, $\lesssim 0.5\,\mathrm{fs}$ at 800 nm). Timing jitter is a per-shot phase factor $e^{i\omega_k\delta t}$.

### Metadata

Save the data with the validated schema:

```python
np.savez_compressed("fields/f0.25.npz",
    Ec=Ec,                                   # (Nf, Nz, Ny, Nx) complex128
    metadata=json.dumps(meta))               # meta conforms to data/schema.json
```

Also validated at load time.

## Development and execution

Package management is done via `pixi`. Each module is independent from each other, then imported into centralized execution notebooks. CLI entrypoints are optional.

## Conventions (global)

- SI units in all data; relativistic amplitudes ($a_0 = eE/mc\omega_0$) only as derived reporting quantities.
- Phasor convention: $E = \mathrm{Re}[E_c\, e^{+i\omega t}]$.
- Grid axes slow-to-fast: `(Nf, Nz, Ny, Nx)`. Electron along $+\hat{y}$; laser incident along $-\hat{y}$; polarization along $\hat{z}$.
- Pulse durations quoted as **intensity FWHM** (the paper's convention); the field-envelope sigma is $\sigma_t = \tau_{\mathrm{FWHM}}/\sqrt{4\ln 2} \approx \tau_{\mathrm{FWHM}}/1.665$. The schema pins this.
