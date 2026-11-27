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
  - **Quantitative degradation (`comparison.py`):** Compares `gaussian_incident_amplitude` against `ideal_incident_amplitude` (`incident_intensity_profiles`, `compare_profiles`, `apodizer_transmission`, `summarize_degradation`), attaching `mse`, `correlation`, and `transmission` under `metadata["degradation"]` in `gaussian` mode.
  - **Configurable laser parameters:** `cli.py` defaults to PW-class FSQED Ti:Sapphire parameters ($\lambda_0 = 0.8\,\mu\mathrm{m}$, $P = 1\,\mathrm{PW}$, $\tau_{\mathrm{FWHM}} = 30\,\mathrm{fs}$, $f = 0.25$, optional `--n-cycles`), while exposing `--lambda0-m`, `--power-w`, `--tau-fwhm-s`, `--w0-over-rmax`, and `--apod-order` so non-FSQED setups (e.g. campus Ti:Sapphire parameters such as $170\,\mathrm{fs}$, $13\,\mathrm{kW}$, $390\text{–}1040\,\mathrm{nm}$) can be modeled directly via CLI flags.
- **Vectorized Debye synthesis & exact $\mathbf{B}_c$:**
  - `focus.focal_field_angular_spectrum()` evaluates the non-paraxial 3D vectorial Debye integral via vectorized ring-factored quadrature (`np.einsum` + outer `ph_y`), accumulating exact $\mathbf{B}_c$ alongside $\mathbf{E}_c$ using $\mathbf{B}_{\mathrm{far}}(\mathbf{n}) = +\frac{1}{c}\,\mathbf{n}\times\mathbf{E}_{\mathrm{far}}(\mathbf{n})$, and calls `align_global_phase(Ec, Bc)` so peak $\mathbf{E}_c$ is real and positive at $t_{\mathrm{ref}} = 0$.
- **Poynting flux & transverse box size:**
  - `focus.compute_checks()` compares focal-plane Poynting flux `p0` (`iy = Ny // 2`, negative for $-\hat{\mathbf{y}}$ propagation) against `power_w` via `abs((-p0) - power_w) / power_w`.
  - `cli.py` defaults to `span_xy_m = 6.4e-6` ($8\lambda_0$), giving `power_conservation_rel_err ~ 1.5%` at $64^3$ (`gaussian` mode) — well below `load_field_pack`'s `max_power_rel_err = 0.10` threshold.
- **Polarization guard (Option A active):**
  - `cli.py` (`generate_field_pack` and `--pol`) fails fast with `NotImplementedError` / `choices=["z-linear"]` whenever `pol != "z-linear"`, preventing silently mislabeled `"circular"` packs. `metadata.py` and `data/schema.json` retain `"circular"` support for Option B below.
- **`outs.json` snapshot:** Refreshed for the current $64^3$ `gaussian` defaults (`mode: "gaussian"`, `power_conservation_rel_err ≈ 0.015`, `degradation` metrics included).

---

## 5. Completed Milestones & Remaining Planned Work

### Completed (`feat/mirrored-gaussian-beam`)
- **DONE — Global Phase Alignment:** `focus.align_global_phase(Ec, Bc)` rotates `(Ec, Bc)` by a single unit scalar from the peak field so `Ec` is real and positive at $t_{\mathrm{ref}} = 0$ (preserving Maxwell's equations and all relative phases). `@pytest.mark.xfail` has been removed from `tests/test_integration.py::test_raw_optics_pack_loads_in_particles`, which passes for both `"gaussian"` and `"ideal"` modes.
- **DONE — Exact Debye `Bc` Reconstruction:** Each plane wave $e^{-ik\mathbf{n}\cdot\mathbf{r}}$ has $\mathbf{k} = -k\mathbf{n}$, so under $e^{+i\omega t}$ Faraday's law gives $\mathbf{B}_{\mathrm{far}}(\mathbf{n}) = +\frac{1}{c}\,\mathbf{n}\times\mathbf{E}_{\mathrm{far}}(\mathbf{n})$ algebraically. `focal_field_angular_spectrum` accumulates exact `Bc` alongside `Ec` inside the Debye quadrature (zero finite-difference dispersion). `magnetic_field_from_curl` is kept as a standalone helper and cross-checked in `tests/test_optics.py::test_curl_b_matches_direct_b`.
- **DONE — Mirror Reflection + Gaussian Pipeline & Degradation Comparison:** Integrated `gaussian.py`, `focus.mirror_reflection_far_field` / `gaussian_far_field` / `ideal_far_field`, `comparison.py`, and dual-mode `cli.py` (`--mode gaussian|ideal`) + `sweep.py` (`ideal_far_field_builder`, `gaussian_far_field_builder`).
- **DONE — Vectorized Debye Quadrature & Full Metadata Validation:** Ring-factored `np.einsum` + outer `ph_y` evaluation in `focus.focal_field_angular_spectrum` and `focal_field_slice`; `metadata.build_metadata` validates the full metadata dict (including collision-checked `extra` keys) against `data/schema.json` before returning.

### Remaining: Circular Polarization — Option B (after LP baseline)
1. Add `pol: Literal["z-linear", "circular"] = "z-linear"` to `bi_dipole_far_field` / `mirror_reflection_far_field` / `focal_field_angular_spectrum` / `focal_field_slice` / `cli.py`:
   - For `"circular"`, superimpose the $\hat{\mathbf{z}}$-polarized and $90^\circ$-rotated $\hat{\mathbf{x}}$-polarized far fields with a $\pm i$ phase shift ($\mathbf{E}_{\mathrm{ff}}^{(z)} + i\,\mathbf{E}_{\mathrm{ff}}^{(x)}$; power normalization automatically supplies the $1/\sqrt{2}$ amplitude scaling).
2. Update `focus.compute_checks(..., pol=...)`:
   - Evaluate `a_max` and `chi_max_per_GeV` from the **real instantaneous vector field at `t_ref_s`** (matching `src.particles.fields._compute_peak_e_at_t_ref`), **not** `np.max(np.abs(Ec))` or complex norm $\sqrt{|E_{c,x}|^2 + |E_{c,z}|^2}$ (which overstates CP's real rotating field amplitude $|\mathbf{E}(0, t)| = E_0/\sqrt{2}$ by $\sqrt{2}$).
   - Scale `a_max_expected` by $1/\sqrt{2}$ for `"circular"` (and optionally include the truncated bi-dipole factor $\sqrt{(8 - (1 + \cos\zeta)^3)/8}$).
3. Remove the `pol != "z-linear"` `NotImplementedError` guard and `choices=["z-linear"]` restriction in `cli.py` to re-enable `--pol circular`.
4. Add CP integration tests verifying `load_field_pack` and `src.particles.qed` CP selection cuts ($\alpha > 0.7\,a_{\max}^{\mathrm{CP}}/\gamma$, reproducing PDF Fig. 2 & Fig. 5(b)).

### Optional / Future Improvements
- **Fig. 4 sweep range:** `sweep.default_f_number_sweep()` stops at $f = 0.25$, whereas PDF Fig. 4 extends down toward $f \to 0$ (opening angles up to $170^\circ$ wrapping past $90^\circ$ into the $+y$ hemisphere). Both `focus.f_number_to_theta_max` and `focus.mirror_reflection_far_field` already support $f < 0.25$ for $\theta < \pi$.
- **Area-weighted MSE in `comparison.py`:** `compare_profiles` currently normalizes intensity profiles with radial area weight $R$ (`np.trapezoid(I * R, R)`), then computes an unweighted `np.mean((Ii - Ir) ** 2)`; weighting the MSE integral by $R\,dR$ over the circular mirror aperture would make the metric directly area-consistent.
- **Pre-mirror FFT propagation / thin-lens phase:** If modeling non-collimated or aberrated beams before the parabolic mirror becomes necessary, re-introduce 2D paraxial angular-spectrum propagation / thin-lens phase in `gaussian.py` feeding a complex `incident_amplitude` callback into `mirror_reflection_far_field`.
