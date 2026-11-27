# AGENTS.md — `fsqed-laser-electron` Handoff & Architecture Guide

## 1. Project Overview & References
Numerical reproduction of Olofsson & Gonoskov, *"Attaining a strong-field QED signal at laser-electron colliders with optimized focusing"*, Phys. Rev. A **106**, 063512 (2022) ([PDF](docs/Olofsson%20and%20Gonoskov%20-%202022%20-%20Attaining%20a%20strong-field%20QED%20signal%20at%20laser-electron%20colliders%20with%20optimized%20focusing.pdf)).

The codebase is split into two independent packages that communicate solely via compressed `.npz` **field packs** validated against [`data/schema.json`](data/schema.json):
- `src/optics/` — non-paraxial focal-field solver, parabolic-mirror Gaussian pipeline, degradation comparison, and aperture sweeps (see [`docs/OPTICS.md`](docs/OPTICS.md)).
- `src/particles/` — relativistic Vay/Boris electron pusher, LCFA QED emission, selection cuts, and $3\sigma$ collider statistics (see [`docs/PARTICLES.md`](docs/PARTICLES.md)).

---

## 2. Development Workflow & Conventions
- **Branch & commits:** Work on `dev` with small, incremental commits and brief conventional commit messages.
- **Environment & tests:** Managed via `pixi` (`numpy >= 2.0`, `pillow >= 10.0`). Run the full suite with:
  ```sh
  pixi run test
  ```
- **Imports:** Always add the repo root to `sys.path` (if needed) and import via `from src.optics import ...` and `from src.particles import ...` (never `from optics import ...`, to avoid dual-module identities).

---

## 3. Global Physics & Data Contract (Non-Negotiable)
- **Units:** All stored arrays, grids, frequencies, fields, and powers are in **SI units** (`m`, `s`, `rad/s`, `V/m`, `T`, `W`). Relativistic amplitude $a_0 = eE/(m_e c\omega_0)$ is only a derived reporting/check quantity.
- **Array layout:** `Ec` and `Bc` have shape `(Nf, 3, Nz, Ny, Nx)` (`complex128`), slow-to-fast spatial axes `(z, y, x)`, and vector components `(x, y, z)` on axis `1`. Spatial grids in `src/optics` use centered even-size axes (`step = span / n`, `start = -(n // 2) * step`).
- **Collision geometry:**
  - Electrons enter along $+\hat{\mathbf{y}}$.
  - Laser propagates along $-\hat{\mathbf{y}}$ (converging from $n_y < 0$ toward the focus at $\mathbf{r} = \mathbf{0}$).
  - For linear polarization (`"z-linear"`), $\mathbf{E} \parallel +\hat{\mathbf{z}}$ and $\mathbf{B} \parallel -\hat{\mathbf{x}}$ so $\mathbf{E} + \mathbf{v}\times\mathbf{B}$ adds constructively (and Poynting flux $S_y < 0$).
- **Phasor & pulse convention:**
  - $\mathbf{E}(\mathbf{r}, t) = g(t+\delta t)\sum_k \mathrm{Re}[\mathbf{E}_c^{(k)}(\mathbf{r})\,e^{+i\omega_k(t+\delta t)}]$. Under $e^{+i\omega t}$, Faraday's law is $\mathbf{B}_c = \frac{i}{\omega}\nabla\times\mathbf{E}_c$. Note that the paper uses $e^{-i\omega t}$ (complex-conjugate phasor).
  - Pulse durations $\tau_{\mathrm{FWHM}}$ are **intensity FWHM**; Gaussian field-envelope sigma is $\sigma_t = \tau_{\mathrm{FWHM}}/\sqrt{4\ln 2}$ (`FWHM_TO_FIELD_SIGMA`).
- **f-number convention:** $f = L/(2 R_{\max})$, $\theta_{\max} = 2\arctan(1/(4f))$ (`f_number_to_theta_max`). $f = 0.25$ collects the full $-\hat{\mathbf{y}}$ hemisphere ($\theta_{\max} = \pi/2$).

---

## 4. Key Architectural Choices & Current State
- **Dual-mode far-field pipeline (`feat/mirrored-gaussian-beam`):**
  - `cli.py` and `sweep.py` now support both the realistic parabolic-mirror Gaussian pipeline (`--mode gaussian`, default) and the ideal Sheppard–Larkin benchmark (`--mode ideal`, `bi_dipole_far_field`).
  - **Realistic mode (`--mode gaussian`):** `gaussian.py` evaluates normalized-radius ($r = R/R_{\max}$) incident profiles (`gaussian_source`, `apodizer`, `gaussian_incident_amplitude`), and `focus.mirror_reflection_far_field` maps $r = \tan(\theta/2)/\tan(\theta_{\max}/2)$, applies local-normal reflection $\mathbf{E}^p = \mathbf{E} - 2(\mathbf{E}\cdot\mathbf{N})\mathbf{N}$ (`mirror_normal`, `reflect`) and the parabolic mirror amplitude Jacobian $2/(1+\cos\theta)$, supporting $f < 0.25$ up to $\theta < \pi$.
  - **Ideal baseline (`--mode ideal`):** In Sec. IV (p. 4, Eq. (13)) of the paper, reflecting a $\hat{\mathbf{z}}$-polarized beam with $I^p(R) \propto [(R/2L)^2 + 1]^{-4}$ off a parabolic mirror produces `bi_dipole_far_field(n)`. `focus.ideal_far_field` runs `ideal_incident_amplitude` through `mirror_reflection_far_field` as a self-consistency check matching `bi_dipole_far_field` up to a constant scale.
  - **Quantitative degradation (`comparison.py`):** Compares `gaussian_incident_amplitude` against `ideal_incident_amplitude` (`incident_intensity_profiles`, `compare_profiles`, `apodizer_transmission`, `summarize_degradation`), attaching `mse`, `rmse`, `relative_rmse`, `correlation`, and `transmission` (apodizer-only; `total_transmission` also includes aperture truncation) under `metadata["degradation"]` in `gaussian` mode.
  - **Configurable laser parameters:** `cli.py` defaults to PW-class FSQED Ti:Sapphire parameters ($\lambda_0 = 0.8\,\mu\mathrm{m}$, $P = 1\,\mathrm{PW}$, $\tau_{\mathrm{FWHM}} \approx 13.34\,\mathrm{fs}$ / 5 cycles, $f = 0.25$, optional `--n-cycles`), while exposing `--lambda0-m`, `--power-w`, `--tau-fwhm-s`, `--w0-over-rmax`, and `--apod-order` so non-FSQED setups (e.g. campus Ti:Sapphire parameters such as $170\,\mathrm{fs}$, $13\,\mathrm{kW}$, $390\text{–}1040\,\mathrm{nm}$) can be modeled directly via CLI flags.
- **Vectorized Debye synthesis & exact $\mathbf{B}_c$:**
  - `focus.focal_field_angular_spectrum()` evaluates the non-paraxial 3D vectorial Debye integral via vectorized ring-factored quadrature (`np.einsum` + outer `ph_y`), accumulating exact $\mathbf{B}_c$ alongside $\mathbf{E}_c$ using $\mathbf{B}_{\mathrm{far}}(\mathbf{n}) = +\frac{1}{c}\,\mathbf{n}\times\mathbf{E}_{\mathrm{far}}(\mathbf{n})$, and calls `align_global_phase(Ec, Bc)` so peak $\mathbf{E}_c$ is real and positive at $t_{\mathrm{ref}} = 0$.
- **Poynting flux & transverse box size:**
  - `focus.compute_checks()` compares focal-plane Poynting flux `p0` (`iy = Ny // 2`, negative for $-\hat{\mathbf{y}}$ propagation) against `power_w` via `abs((-p0) - power_w) / power_w`.
  - `cli.py` defaults to `span_xy_m = 6.4e-6` ($8\lambda_0$), giving `power_conservation_rel_err ~ 1.5%` at $64^3$ (`gaussian` mode) — well below `load_field_pack`'s `max_power_rel_err = 0.10` threshold.
- **Circular polarization (`pol="circular"`):**
  - `focus.py`, `sweep.py`, and `cli.py` support both `"z-linear"` and `"circular"` ($\mathbf{E}_{\mathrm{ff}}^{(z)} + i\,\mathbf{E}_{\mathrm{ff}}^{(x)}$ superposition), with `compute_checks` and `sweep.py` evaluating the real instantaneous rotating field amplitude ($E_0/\sqrt{2}$), matching `src.particles.fields` and `src.particles.qed`.
- **Transmission, area-weighted degradation & Fig. 4 sweep:**
  - `comparison.py` computes circular-disk area-weighted ($R\,dR$) `mse`, `rmse`, `correlation`, `aperture_transmission`, `apodizer_transmission`, and `total_transmission`.
  - `sweep.py` covers `[0.02, 2.0]` in `default_f_number_sweep()` (matching PDF Fig. 4) and accepts `power_transmission` to fold optical losses into `kappa_eff`.
  - `cli.py` adds `--preset {paper,unal}` and `--account-for-losses` (default **on**: an `argparse.BooleanOptionalAction`, so `--no-account-for-losses` restores the lossless behavior; the default `gaussian` pack reports post-loss `power_W`). `presentation_plots.py` renders 7 comparison figures in `images/` (including Paper FSQED and UNAL Ti:Sapphire curves).
- **`outs.json` snapshot:** Refreshed for the current $64^3$ `gaussian` defaults with loss accounting on (`mode: "gaussian"`, `account_for_losses: true` → post-loss `power_W ≈ 9.65e14` W, `power_conservation_rel_err ≈ 0.015`, area-weighted `degradation` metrics including `relative_rmse` + `kappa_eff_focal`/`kappa_eff_effective`, and `a_max_expected` scaled by the truncated bi-dipole aperture factor from the f-number).

---

## 5. Completed Milestones & Optional Future Work

### Completed
- **DONE — Global Phase Alignment:** `focus.align_global_phase(Ec, Bc)` rotates `(Ec, Bc)` by a single unit scalar from the peak field so `Ec` is real and positive at $t_{\mathrm{ref}} = 0$ (preserving Maxwell's equations and all relative phases, preferring $E_z$ on ties for CP parity).
- **DONE — Exact Debye `Bc` Reconstruction:** `focal_field_angular_spectrum` accumulates exact `Bc` alongside `Ec` inside the vectorized ring-factored Debye quadrature (`np.einsum` + outer `ph_y`), cross-checked against `magnetic_field_from_curl`.
- **DONE — Mirror Reflection + Gaussian Pipeline & Degradation Comparison:** Integrated `gaussian.py`, `focus.mirror_reflection_far_field` / `gaussian_far_field` / `ideal_far_field`, area-weighted `comparison.py` (`mse`, `rmse`, `correlation`, aperture/apodizer transmission), and dual-mode `cli.py` + `sweep.py`.
- **DONE — Circular Polarization (Option B):** Implemented `"circular"` across `focus.py`, `sweep.py`, and `cli.py`, verified against `src.particles.fields.load_field_pack` and `src.particles.qed` CP selection cuts in `tests/test_integration.py`.
- **DONE — Fig. 4 Sweep Range & Paper/UNAL Comparison Plots:** Aligned `sweep.default_f_number_sweep()` to `[0.02, 2.0]` and updated `presentation_plots.py` with Paper FSQED vs. UNAL Ti:Sapphire comparisons.

### Optional / Future Improvements
- **Pre-mirror FFT propagation / thin-lens phase:** Intentionally omitted while treating the beam incident on the parabolic mirror as collimated; if modeling non-collimated or aberrated beams becomes necessary later, 2D paraxial angular-spectrum propagation / thin-lens phase can feed a complex `incident_amplitude` callback into `mirror_reflection_far_field`.
