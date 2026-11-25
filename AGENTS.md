# AGENTS.md — `fsqed-laser-electron` Handoff & Architecture Guide

## 1. Project Overview & References
Numerical reproduction of Olofsson & Gonoskov, *"Attaining a strong-field QED signal at laser-electron colliders with optimized focusing"*, Phys. Rev. A **106**, 063512 (2022) ([PDF](docs/Olofsson%20and%20Gonoskov%20-%202022%20-%20Attaining%20a%20strong-field%20QED%20signal%20at%20laser-electron%20colliders%20with%20optimized%20focusing.pdf)).

The codebase is split into two independent packages that communicate solely via compressed `.npz` **field packs** validated against [`data/schema.json`](data/schema.json):
- `src/optics/` — non-paraxial focal-field solver and aperture sweeps (see [`docs/OPTICS.md`](docs/OPTICS.md)).
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
- **Why `src/optics` starts from `bi_dipole_far_field(n)`:**
  - In Sec. IV (p. 4, Eq. (13)) of the paper, Olofsson & Gonoskov prove analytically via $\mathbf{E}^p = \mathbf{E} - 2(\mathbf{E}\cdot\mathbf{N})\mathbf{N}$ that reflecting a $\hat{\mathbf{z}}$-polarized beam with the Sheppard–Larkin radial intensity profile $I^p(R) \propto [(R/2L)^2 + 1]^{-4}$ off a parabolic mirror produces the exact `bi_dipole_far_field(n)` truncated at $\theta \le \theta_{\max}(f)$.
  - `focus.focal_field_angular_spectrum()` then evaluates the **non-paraxial 3D vectorial Debye integral** over $(\theta, \phi)$ to synthesize the diffracted 3D focal field $\mathbf{E}_c(x, y, z)$. Unlike `src.particles.fields.make_analytic_bidipole_pack()` (which only rescales the amplitude of an ideal $4\pi$ spherical-Bessel dipole), `src/optics` captures the actual finite-$f$-number 3D diffraction geometry (transverse waist, Rayleigh length, aperture fringes).
- **Standalone / unwired helpers:**
  - `src/optics/gaussian.py` (Gaussian source, thin lens, apodizer, 2D paraxial FFT propagation, power transmission) and `focus.mirror_normal` / `focus.reflect` are standalone helpers implementing the extended paraxial shaping pipeline from `README.md`. They are **not wired** into `cli.py` or `sweep.py` because the PDF never models a spatial Gaussian beam or lens; only wire them in if extending *beyond* the paper via a custom `far_field(n)` callback.
- **Poynting flux & transverse box size:**
  - `focus.compute_checks()` uses `abs(p0 - p1) / abs(p0)` (`abs(p0) > 0.0`) because $S_y < 0$ for $-\hat{\mathbf{y}}$ propagation.
  - `cli.py` defaults to `span_xy_m = 6.4e-6` ($8\lambda_0$) so lateral flux escape for $f/0.25$ stays below `load_field_pack`'s `max_power_rel_err = 0.10` threshold (`~8.3%` at $64^3$; vs. `~18.4%` at `4.0e-6`).
- **Polarization guard (Option A active):**
  - `cli.py` (`generate_field_pack` and `--pol`) fails fast with `NotImplementedError` whenever `pol != "z-linear"`, preventing silently mislabeled `"circular"` packs. `metadata.py` and `data/schema.json` retain `"circular"` support for Option B below.
- **Stale snapshot:** `outs.json` at the repo root predates the `span_xy_m` / `power_conservation_rel_err` fixes and should be regenerated once baseline optics fixes settle.

---

## 5. Pending Decisions & Planned Work (In Order)

### 1. Global Phase Alignment (`-1j` Gouy phase blocker — needed for LP baseline)
- **Current state:** `focus.focal_field_angular_spectrum` (and `focal_field_slice`, `sweep.focal_field_vector`) uses `pref = -1j * Cscale / lambda0_m`. For real `bi_dipole_far_field`, `Ec(0)` is purely imaginary (`Re[Ec(0)] = 0`), whereas `src.particles.fields.load_field_pack` validates `checks.a_max` against $\max_{\mathbf{r}}|\mathrm{Re}[\mathbf{E}_c(\mathbf{r}) e^{i\omega t_{\mathrm{ref}}}]|$ at `t_ref_s = 0.0`. As a result, `load_field_pack` fails (`rel_diff = 19.3% > 5%`).
- **Pinned test:** `tests/test_integration.py::test_raw_optics_pack_loads_in_particles` is currently `@pytest.mark.xfail(strict=True, raises=ValueError, ...)`, while `test_phase_aligned_optics_pack_propagates_in_particles` verifies end-to-end propagation after multiplying `(Ec, Bc)` by `-1j`.
- **Preferred fix:** Phase-align `(Ec, Bc)` by a single global unit scalar $e^{-i\phi_0}$ (chosen from the peak field at focus so `Ec` at peak is real and positive at `t_ref_s = 0`).
  - Matches PDF Sec. III Eqs. (6)–(7) *"synchronous peaking"* at $t = 0$.
  - Preserves Maxwell's equations and all relative phases ($\mathbf{E}$–$\mathbf{B}$ and CP's $\pi/2$ phase).
  - Remains valid even if `far_field(n)` is complex (e.g., if `gaussian.py` + `thin_lens_phase` + `reflect` are later wired in).
  - Once landed: remove `@pytest.mark.xfail` from `test_raw_optics_pack_loads_in_particles`, regenerate `outs.json`, and run LP baseline comparisons against `make_analytic_bidipole_pack` and PDF Fig. 5(a).

### 2. Circular Polarization — Option B (after LP baseline)
1. Add `pol: Literal["z-linear", "circular"] = "z-linear"` to `bi_dipole_far_field` / `focal_field_angular_spectrum` / `focal_field_slice` / `cli.py`:
   - For `"circular"`, superimpose the $\hat{\mathbf{z}}$-polarized and $90^\circ$-rotated $\hat{\mathbf{x}}$-polarized bi-dipole far fields with a $\pm i$ phase shift ($\mathbf{E}_{\mathrm{ff}}^{(z)} + i\,\mathbf{E}_{\mathrm{ff}}^{(x)}$; power normalization automatically supplies the $1/\sqrt{2}$ amplitude scaling).
2. Update `focus.compute_checks(..., pol=...)`:
   - Evaluate `a_max` and `chi_max_per_GeV` from the **real instantaneous vector field at `t_ref_s`** (matching `src.particles.fields._compute_peak_e_at_t_ref`), **not** `np.max(np.abs(Ec))` or complex norm $\sqrt{|E_{c,x}|^2 + |E_{c,z}|^2}$ (which overstates CP's real rotating field amplitude $|\mathbf{E}(0, t)| = E_0/\sqrt{2}$ by $\sqrt{2}$).
   - Scale `a_max_expected` by $1/\sqrt{2}$ for `"circular"` (and optionally include the truncated bi-dipole factor $\sqrt{(8 - (1 + \cos\zeta)^3)/8}$).
3. Remove the `pol != "z-linear"` `NotImplementedError` guard in `cli.py` and re-enable `--pol circular`.
4. Add CP integration tests verifying `load_field_pack` and `src.particles.qed` CP selection cuts ($\alpha > 0.7\,a_{\max}^{\mathrm{CP}}/\gamma$, reproducing PDF Fig. 2 & Fig. 5(b)).

### 3. Exact Debye `Bc` Reconstruction (accuracy upgrade after baseline results)
- **Current state:** `focus.focal_field_angular_spectrum` reconstructs `Bc = (i/omega) curl(Ec)` via 2nd-order central finite differences (`focus.magnetic_field_from_curl` using `np.gradient`). This underestimates $|B_{c,x}(0)|$ by **~2.3%** at $64^3$ ($\Delta y = 62.5\,\mathrm{nm}$) and **~9%** at $32^3$, and incurs 1st-order boundary errors.
- **Upgrade:** Each plane wave in the Debye sum has $\mathbf{k} = -k\mathbf{n}$, so Faraday's law gives $\mathbf{B}_{\mathrm{far}}(\mathbf{n}) = -\frac{1}{c}\,\mathbf{n}\times\mathbf{E}_{\mathrm{far}}(\mathbf{n})$ algebraically. Accumulate `Bc` directly inside the Debye quadrature loop using the same `phase` array already computed for `Ec` (exact to machine precision, zero finite-difference dispersion). Keep `magnetic_field_from_curl` as a public helper/fallback.

### 4. Optional / Future Improvements
- **Fig. 4 sweep range:** `sweep.default_f_number_sweep()` stops at $f = 0.25$, whereas PDF Fig. 4 extends down toward $f \to 0$ (opening angles up to $170^\circ$ wrapping past $90^\circ$ into the $+y$ hemisphere). `focus.f_number_to_theta_max` already supports $f < 0.25$.
- **Realistic pre-mirror beam pipeline:** Wire `gaussian.py` + `focus.mirror_normal` / `focus.reflect` (including the parabolic surface Jacobian $\propto [(R/2L)^2 + 1]^{-1}$) into a `far_field(n)` factory if comparing lab Gaussian beams against the ideal Sheppard–Larkin profile.
- **Performance:** Vectorize or Numba-JIT the nested `(ntheta, nphi)` loops in `focus.py` for faster $128^3+$ grids.
