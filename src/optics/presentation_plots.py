"""Generate presentation and comparison figures for the optics simulation.

Uses ``matplotlib`` (``Agg`` backend) and ``numpy``.  Run with:

    pixi run python src/optics/presentation_plots.py

Generates 7 PNG figures in ``images/``, including side-by-side comparisons of the
ideal Sheppard-Larkin baseline (``bi_dipole_far_field``) against the realistic
Gaussian + apodizer + parabolic-mirror pipeline across the PDF Fig. 4 f-number range
(``0.02 .. 2.0``) for both the Paper FSQED laser and UNAL's Ti:Sapphire laser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optics import comparison  # noqa: E402
from src.optics import constants as C  # noqa: E402
from src.optics import focus  # noqa: E402
from src.optics import sweep  # noqa: E402

OUT = REPO_ROOT / "images"
UM = 1.0e6
FN_REF = 0.25
W0_OVER_RMAX = 0.7
APOD_ORDER = 8.0


def _field_intensity_slices(
    *,
    lambda0_m: float,
    power_w: float,
    far_field,
    f_number: float = FN_REF,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute focal (y=0) and convergence (x=0) |E|^2 slices for a given far-field."""
    Ec_xz, ax_x, ax_z, _ = focus.focal_field_slice(
        plane="y",
        coord_m=0.0,
        lambda0_m=lambda0_m,
        f_number=f_number,
        power_w=power_w,
        n1=240,
        n2=240,
        span1_m=3.0e-6,
        span2_m=3.0e-6,
        ntheta=48,
        nphi=96,
        far_field=far_field,
    )
    I_xz = np.sum(np.abs(Ec_xz) ** 2, axis=0)

    Ec_yz, ax_z2, ax_y, _ = focus.focal_field_slice(
        plane="x",
        coord_m=0.0,
        lambda0_m=lambda0_m,
        f_number=f_number,
        power_w=power_w,
        n1=240,
        n2=240,
        span1_m=3.0e-6,
        span2_m=6.0e-6,
        ntheta=48,
        nphi=96,
        far_field=far_field,
    )
    I_yz = np.sum(np.abs(Ec_yz) ** 2, axis=0)
    return (I_xz, ax_x, ax_z), (I_yz, ax_z2, ax_y)


def _add_opening_angle_top_axis(ax: plt.Axes) -> plt.Axes:
    """Attach a secondary top x-axis with Fig. 4 opening-angle tick marks."""
    angles_deg = [170, 120, 80, 50, 30, 20, 15]
    f_positions = [1.0 / (4.0 * np.tan(np.radians(a) / 2.0)) for a in angles_deg]
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(f_positions)
    ax_top.set_xticklabels([f"{a}°" for a in angles_deg], fontsize=9)
    ax_top.set_xlabel("opening angle $\\zeta$", fontsize=10, labelpad=6)
    return ax_top


def _add_field_right_axis(
    ax: plt.Axes,
    *,
    energy_gev: float,
    scale_exponent: float,
    ylabel: str,
) -> plt.Axes:
    """Attach a right y-axis converting chi_max (at 1 GeV) to peak |E|."""
    gamma = C.energy_gev_to_gamma(energy_gev)
    chi_to_e = C.E_CRIT / (2.0 * gamma) / (10.0 ** scale_exponent)
    ax_right = ax.twinx()
    y0, y1 = ax.get_ylim()
    ax_right.set_ylim(y0 * chi_to_e, y1 * chi_to_e)
    ax_right.set_ylabel(ylabel, fontsize=10)
    return ax_right


def _run_fig4_sweeps(
    fns: np.ndarray,
    *,
    lambda0_m: float,
    power_w: float,
    energy_gev: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run ideal, lossless-gaussian, and lossy-gaussian f-number sweeps."""
    t_total = sweep.gaussian_transmission(W0_OVER_RMAX, APOD_ORDER)
    ideal = sweep.kappa_eff_vs_f_number(
        fns,
        lambda0_m=lambda0_m,
        power_w=power_w,
        energy_gev=energy_gev,
        far_field_builder=sweep.ideal_far_field_builder(),
        ntheta=192,
        nphi=256,
    )
    gauss_focal = sweep.kappa_eff_vs_f_number(
        fns,
        lambda0_m=lambda0_m,
        power_w=power_w,
        energy_gev=energy_gev,
        far_field_builder=sweep.gaussian_far_field_builder(W0_OVER_RMAX, APOD_ORDER),
        power_transmission=1.0,
        ntheta=192,
        nphi=256,
    )
    gauss_loss = sweep.kappa_eff_vs_f_number(
        fns,
        lambda0_m=lambda0_m,
        power_w=power_w,
        energy_gev=energy_gev,
        far_field_builder=sweep.gaussian_far_field_builder(W0_OVER_RMAX, APOD_ORDER),
        power_transmission=t_total,
        ntheta=192,
        nphi=256,
    )
    return ideal, gauss_focal, gauss_loss


def image_focal_intensity(out: Path, ideal_slices, gauss_slices) -> None:
    """01: Side-by-side focal plane (y = 0) |E|^2 comparison."""
    (I_id, ax_x, ax_z), _ = ideal_slices
    (I_ga, _, _), _ = gauss_slices
    log_id = np.log10(I_id + 1e-30)
    log_ga = np.log10(I_ga + 1e-30)
    vmax = float(max(log_id.max(), log_ga.max()))
    vmin = vmax - 6.0
    extent = [ax_x[0] * UM, ax_x[-1] * UM, ax_z[0] * UM, ax_z[-1] * UM]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, arr, subtitle in (
        (axes[0], log_id, "Ideal (Sheppard-Larkin)"),
        (axes[1], log_ga, f"Realista Gaussiano ($w_0/R_{{\\max}}={W0_OVER_RMAX}$, $m={APOD_ORDER:.0f}$)"),
    ):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="inferno", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("z (µm)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label(r"$\log_{10} |E|^2\ (\mathrm{V^2/m^2})$")
    fig.suptitle(r"Campo focal $|E|^2$ en el plano $y=0$ ($P=1\,\mathrm{PW}$, $\lambda=0.8\,\mu\mathrm{m}$, $f=0.25$)", fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def image_convergence(out: Path, ideal_slices, gauss_slices) -> None:
    """02: Side-by-side convergence plane (x = 0) |E|^2 comparison."""
    _, (I_id, ax_z, ax_y) = ideal_slices
    _, (I_ga, _, _) = gauss_slices
    log_id = np.log10(I_id + 1e-30)
    log_ga = np.log10(I_ga + 1e-30)
    vmax = float(max(log_id.max(), log_ga.max()))
    vmin = vmax - 6.0
    extent = [ax_z[0] * UM, ax_z[-1] * UM, ax_y[0] * UM, ax_y[-1] * UM]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, arr, subtitle in (
        (axes[0], log_id, "Ideal (Sheppard-Larkin)"),
        (axes[1], log_ga, f"Realista Gaussiano ($w_0/R_{{\\max}}={W0_OVER_RMAX}$, $m={APOD_ORDER:.0f}$)"),
    ):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="inferno", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel("z (µm)")
        ax.set_ylabel("y (µm)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label(r"$\log_{10} |E|^2\ (\mathrm{V^2/m^2})$")
    fig.suptitle(r"Convergencia del haz en el plano $x=0$ ($P=1\,\mathrm{PW}$, $\lambda=0.8\,\mu\mathrm{m}$, $f=0.25$)", fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def image_profile(out: Path, ideal_slices, gauss_slices) -> None:
    """03: Focal field cross-sections |E(x,0,0)| and |E(0,y,0)| for Paper & UNAL lasers."""
    (I_id_xz, ax_x, _), (I_id_yz, _, ax_y) = ideal_slices
    (I_ga_xz, _, _), (I_ga_yz, _, _) = gauss_slices

    # UNAL fields scale by sqrt(P_unal / P_paper) at the same wavelength and f-number.
    unal_field_scale = np.sqrt(C.UNAL_TISAPPHIRE_LASER["power_w"] / C.PAPER_FSQED_LASER["power_w"])
    t_total = sweep.gaussian_transmission(W0_OVER_RMAX, APOD_ORDER)
    loss_field_scale = np.sqrt(t_total)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    configs = [
        (axes[0], 1.0, 1e14, "Paper FSQED (1 PW, 0.8 µm, 30 fs)", r"$|E|\ (10^{14}\ \mathrm{V/m})$"),
        (axes[1], unal_field_scale, 1e9, "UNAL Ti:Sapphire (13 kW, 0.8 µm, 170 fs)", r"$|E|\ (10^{9}\ \mathrm{V/m})$"),
    ]
    for ax, fscale, norm, title, ylabel in configs:
        ex_id = np.sqrt(I_id_xz[I_id_xz.shape[0] // 2, :]) * fscale / norm
        ey_id = np.sqrt(I_id_yz[:, I_id_yz.shape[1] // 2]) * fscale / norm
        ex_ga = np.sqrt(I_ga_xz[I_ga_xz.shape[0] // 2, :]) * fscale * loss_field_scale / norm
        ey_ga = np.sqrt(I_ga_yz[:, I_ga_yz.shape[1] // 2]) * fscale * loss_field_scale / norm

        ax.plot(ax_x * UM, ex_id, color="C0", lw=2.0, label="Ideal vs x (transversal)")
        ax.plot(ax_y * UM, ey_id, color="C0", lw=1.6, ls="--", label="Ideal vs y (longitudinal)")
        ax.plot(ax_x * UM, ex_ga, color="C3", lw=2.0, label="Gaussiano + T vs x")
        ax.plot(ax_y * UM, ey_ga, color="C3", lw=1.6, ls="--", label="Gaussiano + T vs y")
        ax.set_xlim(-1.5, 1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("posición (µm)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Perfil espacial del campo eléctrico en el foco ($f = 0.25$): Paper vs. UNAL", fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def image_kappa_sweep(out: Path) -> None:
    """04: Reproduction of PDF Fig. 4 across f-number in [0.02, 2.0]."""
    fns = sweep.default_f_number_sweep()
    ideal, gauss_focal, gauss_loss = _run_fig4_sweeps(fns, **C.PAPER_FIG4_LASER)
    f = np.array([r["f_number"] for r in ideal])
    chi_id = np.array([r["chi_max_per_GeV"] for r in ideal])
    chi_gf = np.array([r["chi_max_per_GeV"] for r in gauss_focal])
    chi_gl = np.array([r["chi_max_per_GeV"] for r in gauss_loss])
    t_total = gauss_loss[0]["transmission"]

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    ax.plot(f, chi_id, "o-", color="C0", ms=4, lw=2.0, label="Ideal (Sheppard-Larkin)")
    ax.plot(f, chi_gf, "--", color="C1", lw=1.8, label="Gaussiano realista (potencia en foco)")
    ax.plot(f, chi_gl, "s-", color="C3", ms=3.8, lw=2.0, label=f"Gaussiano realista (con pérdida $T={t_total:.3f}$)")
    ax.axhline(C.KAPPA_BIDIPOLE_MAX, color="0.35", ls=":", lw=1.4, label=r"Límite bi-dipolo $\kappa_{\max} = 5.25$")

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 5.8)
    ax.set_xlabel("f-number  $L / (2 R_{\\max})$", fontsize=11)
    ax.set_ylabel(r"$\chi_{\max}$ (o $\kappa_{\mathrm{eff}}$ para $1\ \mathrm{PW}, 1\ \mu\mathrm{m}, 1\ \mathrm{GeV}$)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    _add_opening_angle_top_axis(ax)
    _add_field_right_axis(
        ax,
        energy_gev=C.PAPER_FIG4_LASER["energy_gev"],
        scale_exponent=14,
        ylabel=r"field strength ($10^{14}\ \mathrm{V/m}$)",
    )
    ax.set_title("Reproducción Fig. 4: $\\chi_{\\max}$ vs. f-number (ideal vs. gaussiano realista)", fontsize=11.5, fontweight="bold", pad=10)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def image_farfield(out: Path) -> None:
    """05: Azimuthally averaged far-field angular distribution at f = 0.25."""
    th = np.linspace(0.0, np.pi / 2.0, 300)
    ph = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(TH) * np.cos(PH), -np.cos(TH), np.sin(TH) * np.sin(PH)], axis=-1)

    E_id = focus.bi_dipole_far_field(n)
    E_ga = focus.gaussian_far_field(n, f_number=FN_REF, w0_over_rmax=W0_OVER_RMAX, apod_order=APOD_ORDER)
    I_id = np.mean(np.sum(np.abs(E_id) ** 2, axis=-1), axis=1)
    I_ga = np.mean(np.sum(np.abs(E_ga) ** 2, axis=-1), axis=1)
    I_id /= I_id[0]
    I_ga /= I_ga[0]

    th_deg = np.degrees(th)
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    ax.plot(th_deg, I_id, color="C0", lw=2.2, label="Ideal (Sheppard-Larkin)")
    ax.plot(th_deg, I_ga, color="C3", lw=2.2, label=f"Gaussiano + espejo ($w_0/R_{{\\max}}={W0_OVER_RMAX}$, $m={APOD_ORDER:.0f}$)")
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$\theta$ desde el eje $-\hat{y}$ (grados)", fontsize=11)
    ax.set_ylabel(r"$\langle |E_{\mathrm{ff}}(\theta)|^2 \rangle_\phi$ normalizado", fontsize=11)
    ax.set_title("Distribución angular del campo lejano tras el espejo parabólico ($f = 0.25$)", fontsize=11.5, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9.5)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def image_incident_profile(out: Path) -> None:
    """06: Radial incident intensity profile I^p(R) (Eq. 13) vs Gaussian x apodizer."""
    R, I_id, I_ga = comparison.incident_intensity_profiles(
        f_number=FN_REF, w0_over_rmax=W0_OVER_RMAX, apod_order=APOD_ORDER, nr=512
    )
    metrics = comparison.summarize_degradation(
        f_number=FN_REF, w0_over_rmax=W0_OVER_RMAX, apod_order=APOD_ORDER, nr=512
    )
    I_id_n = I_id / I_id[0]
    I_ga_n = I_ga / I_ga[0]

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    ax.plot(R, I_id_n, color="C0", lw=2.3, label=r"Ideal Ec. (13): $[(R/2L)^2 + 1]^{-4}$")
    ax.plot(R, I_ga_n, color="C3", lw=2.3, ls="--", label=f"Gaussiano $\\times$ apodizador ($w_0/R_{{\\max}}={W0_OVER_RMAX}$, $m={APOD_ORDER:.0f}$)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Radio normalizado $r = R / R_{\max}$", fontsize=11)
    ax.set_ylabel(r"Intensidad incidente $I^p(r) / I^p(0)$", fontsize=11)
    ax.set_title("Perfil radial incidente antes del espejo parabólico ($f = 0.25$)", fontsize=11.5, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9.5)

    note = (
        f"Métricas ponderadas por área ($R\\,dR$):\n"
        f"RMSE = {metrics['rmse']:.4f}   (MSE = {metrics['mse']:.4f})\n"
        f"Correlación r = {metrics['correlation']:.6f}\n"
        f"T_apodizador = {metrics['transmission']:.4f}\n"
        f"T_total = {metrics['total_transmission']:.4f}"
    )
    ax.text(
        0.97, 0.96, note,
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.7", alpha=0.92),
    )
    fig.savefig(out, dpi=180)
    plt.close(fig)


def image_unal_vs_paper_comparison(out: Path) -> None:
    """07: Paper FSQED vs UNAL Ti:Sapphire sweep across Fig. 4 range (0.02 .. 2.0)."""
    fns = sweep.default_f_number_sweep()
    paper_id, paper_gf, paper_gl = _run_fig4_sweeps(
        fns,
        lambda0_m=C.PAPER_FSQED_LASER["lambda0_m"],
        power_w=C.PAPER_FSQED_LASER["power_w"],
        energy_gev=C.PAPER_FSQED_LASER["energy_gev"],
    )
    unal_id, unal_gf, unal_gl = _run_fig4_sweeps(
        fns,
        lambda0_m=C.UNAL_TISAPPHIRE_LASER["lambda0_m"],
        power_w=C.UNAL_TISAPPHIRE_LASER["power_w"],
        energy_gev=C.UNAL_TISAPPHIRE_LASER["energy_gev"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    panels = [
        (
            axes[0],
            paper_id, paper_gf, paper_gl,
            1.0, 14,
            "Paper FSQED ($1\\ \\mathrm{PW},\\ 0.8\\ \\mu\\mathrm{m},\\ 30\\ \\mathrm{fs}$)",
            r"$\chi_{\max}\ /\ \mathrm{GeV}$",
            r"$|E_{\max}|\ (10^{14}\ \mathrm{V/m})$",
        ),
        (
            axes[1],
            unal_id, unal_gf, unal_gl,
            1.0e5, 9,
            "UNAL Ti:Sapphire ($13\\ \\mathrm{kW},\\ 0.8\\ \\mu\\mathrm{m},\\ 170\\ \\mathrm{fs}$)",
            r"$\chi_{\max}\ /\ \mathrm{GeV}\ (\times 10^{-5})$",
            r"$|E_{\max}|\ (10^{9}\ \mathrm{V/m})$",
        ),
    ]
    for ax, r_id, r_gf, r_gl, yscale, exp_e, title, ylabel, rylabel in panels:
        f = np.array([r["f_number"] for r in r_id])
        y_id = np.array([r["chi_max_per_GeV"] for r in r_id]) * yscale
        y_gf = np.array([r["chi_max_per_GeV"] for r in r_gf]) * yscale
        y_gl = np.array([r["chi_max_per_GeV"] for r in r_gl]) * yscale

        ax.plot(f, y_id, "o-", color="C0", ms=3.5, lw=2.0, label="Ideal (Sheppard-Larkin)")
        ax.plot(f, y_gf, "--", color="C1", lw=1.8, label="Gaussiano (en foco)")
        ax.plot(f, y_gl, "s-", color="C3", ms=3.5, lw=2.0, label="Gaussiano (con pérdida $T$)")
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(0.0, float(y_id.max()) * 1.12)
        ax.set_xlabel("f-number  $L / (2 R_{\\max})$", fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8.5)
        _add_opening_angle_top_axis(ax)

        # Right axis in physical V/m units matching yscale.
        gamma = C.energy_gev_to_gamma(1.0)
        chi_to_e = C.E_CRIT / (2.0 * gamma) / yscale / (10.0 ** exp_e)
        ax_r = ax.twinx()
        y0, y1 = ax.get_ylim()
        ax_r.set_ylim(y0 * chi_to_e, y1 * chi_to_e)
        ax_r.set_ylabel(rylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    fig.suptitle("Comparación Sheppard-Larkin vs. Gaussiano realista: Paper vs. Láser UNAL", fontsize=12.5, fontweight="bold")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lam = C.PAPER_FSQED_LASER["lambda0_m"]
    pwr = C.PAPER_FSQED_LASER["power_w"]
    print("Calculando cortes del campo focal (ideal y gaussiano)...")
    ideal_slices = _field_intensity_slices(lambda0_m=lam, power_w=pwr, far_field=focus.bi_dipole_far_field)
    gauss_slices = _field_intensity_slices(
        lambda0_m=lam,
        power_w=pwr,
        far_field=lambda n: focus.gaussian_far_field(n, f_number=FN_REF, w0_over_rmax=W0_OVER_RMAX, apod_order=APOD_ORDER),
    )

    jobs = [
        (OUT / "01_campo_focal_xz.png", lambda p: image_focal_intensity(p, ideal_slices, gauss_slices)),
        (OUT / "02_convergencia_yz.png", lambda p: image_convergence(p, ideal_slices, gauss_slices)),
        (OUT / "03_perfil_campo.png", lambda p: image_profile(p, ideal_slices, gauss_slices)),
        (OUT / "04_chi_vs_fnumber.png", image_kappa_sweep),
        (OUT / "05_patron_angular.png", image_farfield),
        (OUT / "06_perfil_incidente_eq13.png", image_incident_profile),
        (OUT / "07_unal_vs_paper_comparison.png", image_unal_vs_paper_comparison),
    ]
    for path, fn in jobs:
        print(f"Generando {path.name} ...")
        fn(path)
    print(f"\nListo. {len(jobs)} imágenes en {OUT.resolve()}")


if __name__ == "__main__":
    main()
