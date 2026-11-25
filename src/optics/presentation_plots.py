"""Genera imágenes para la presentación de la simulación óptica bi-dipolo.

Usa Pillow (PIL) + numpy; no requiere matplotlib.  Ejecutar:

    python src/optics/presentation_plots.py

Genera archivos PNG en la carpeta ``images/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optics import constants as C  # noqa: E402
from optics import focus  # noqa: E402
from optics import sweep  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "images"

# --------------------------------------------------------------------------- #
# Paleta de color (aproximación de "inferno", perceptualmente uniforme).
# --------------------------------------------------------------------------- #
INFERNO = [
    (0.00, (0, 0, 4)),
    (0.13, (28, 16, 68)),
    (0.25, (79, 18, 123)),
    (0.38, (135, 37, 129)),
    (0.50, (189, 55, 99)),
    (0.63, (232, 89, 63)),
    (0.75, (250, 135, 40)),
    (0.87, (252, 197, 47)),
    (1.00, (252, 255, 164)),
]
VIRIDIS = [
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
]


def colorize(z: np.ndarray, cmap, vmin: float, vmax: float) -> np.ndarray:
    """Map values in ``[vmin, vmax]`` to an (H, W, 3) uint8 RGB array."""
    t = np.clip((z - vmin) / (vmax - vmin), 0.0, 1.0)
    pos = np.array([p for p, _ in cmap])
    cols = np.array([c for _, c in cmap], dtype=float)
    idx = np.searchsorted(pos, t)
    idx = np.clip(idx, 1, len(pos) - 1)
    lo = pos[idx - 1]
    hi = pos[idx]
    c0 = cols[idx - 1]
    c1 = cols[idx]
    f = ((t - lo) / np.where(hi > lo, hi - lo, 1.0))[..., None]
    rgb = c0 * (1.0 - f) + c1 * f
    return rgb.astype(np.uint8)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("arialbd.ttf" if bold else "arial.ttf"), ("segoeuib.ttf" if bold else "segoeui.ttf")):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _nice_step(span: float, n: int = 5) -> float:
    raw = span / n
    if raw <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def nice_ticks(lo: float, hi: float, n: int = 5) -> np.ndarray:
    step = _nice_step(hi - lo, n)
    start = np.ceil(lo / step) * step
    return np.arange(start, hi + 1e-9 * step, step)


def _fmt(v: float) -> str:
    if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
        return f"{v:.1e}"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}"


def draw_heatmap(
    path: Path,
    z: np.ndarray,
    x_extent,
    y_extent,
    title: str,
    xlabel: str,
    ylabel: str,
    clabel: str,
    vmin: float,
    vmax: float,
    cmap=INFERNO,
    size=(1500, 1100),
) -> None:
    """Render a labelled 2-D heatmap (z[0, :] at y_extent[0]) to a PNG."""
    W, H = size
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = _font(44, bold=True)
    f_axis = _font(34)
    f_tick = _font(26)
    f_cb = _font(30)

    d.text((W // 2, 46), title, font=f_title, fill="black", anchor="ma")

    m_left, m_top, m_bottom = 130, 130, 150
    cb_w, cb_gap = 46, 40
    plot_right = W - 190 - cb_w - cb_gap
    plot_w = plot_right - m_left
    plot_h = H - m_top - m_bottom

    rgb = colorize(z[::-1, :], cmap, vmin, vmax)  # flip so y_extent[1] is on top
    im = Image.fromarray(rgb).resize((plot_w, plot_h), Image.BILINEAR)
    img.paste(im, (m_left, m_top))

    # Colorbar (top = vmax).
    grad = np.linspace(1.0, 0.0, plot_h)[:, None]
    gb = np.repeat(colorize(grad, cmap, 0.0, 1.0), cb_w // 2, axis=1)
    cb_x = plot_right + cb_gap
    img.paste(Image.fromarray(gb).resize((cb_w // 2, plot_h)), (cb_x, m_top))
    d.rectangle([cb_x, m_top, cb_x + cb_w // 2 - 1, m_top + plot_h - 1], outline="black")
    for i in range(6):
        val = vmax - (vmax - vmin) * i / 5
        yp = m_top + plot_h * i / 5
        d.line([cb_x + cb_w // 2 - 1, yp, cb_x + cb_w // 2 + 6, yp], fill="black", width=2)
        d.text((cb_x + cb_w // 2 + 12, yp), _fmt(val), font=f_cb, fill="black", anchor="lm")
    d.text((cb_x + cb_w // 4, m_top + plot_h + 18), clabel, font=f_tick, fill="black", anchor="ma")

    # Plot border.
    d.rectangle([m_left, m_top, m_left + plot_w - 1, m_top + plot_h - 1], outline="black")

    # X ticks.
    x0, x1 = x_extent
    for t in nice_ticks(x0, x1, 5):
        xp = m_left + (t - x0) / (x1 - x0) * (plot_w - 1)
        d.line([xp, m_top + plot_h - 1, xp, m_top + plot_h + 8], fill="black", width=2)
        d.text((xp, m_top + plot_h + 16), _fmt(t), font=f_tick, fill="black", anchor="ma")
    d.text((m_left + plot_w / 2, m_top + plot_h + 58), xlabel, font=f_axis, fill="black", anchor="ma")

    # Y ticks.
    y0, y1 = y_extent
    for t in nice_ticks(y0, y1, 5):
        yp = m_top + (1 - (t - y0) / (y1 - y0)) * (plot_h - 1)
        d.line([m_left - 8, yp, m_left - 1, yp], fill="black", width=2)
        d.text((m_left - 14, yp), _fmt(t), font=f_tick, fill="black", anchor="rm")
    d.text((m_left - 78, m_top + plot_h / 2), ylabel, font=f_axis, fill="black", anchor="mm")

    img.save(path)


def draw_line_plot(
    path: Path,
    series,
    title: str,
    xlabel: str,
    ylabel: str,
    xlog: bool = False,
    hlines=None,
    x_extent=None,
    y_extent=None,
    size=(1500, 1100),
) -> None:
    """Render a labelled line plot (one or several series) to a PNG."""
    W, H = size
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = _font(44, bold=True)
    f_axis = _font(34)
    f_tick = _font(26)
    f_legend = _font(28)

    d.text((W // 2, 46), title, font=f_title, fill="black", anchor="ma")

    m_left, m_top, m_bottom, m_right = 140, 120, 150, 100
    plot_w = W - m_left - m_right
    plot_h = H - m_top - m_bottom

    allx = np.concatenate([np.atleast_1d(s["x"]) for s in series])
    ally = np.concatenate(
        [np.atleast_1d(s["y"]) for s in series]
        + [np.atleast_1d(h["y"]) for h in (hlines or [])]
    )
    if x_extent is None:
        x0, x1 = float(allx.min()), float(allx.max())
        if x0 == x1:
            x0, x1 = x0 - 1, x1 + 1
    else:
        x0, x1 = x_extent
    if y_extent is None:
        y0, y1 = float(ally.min()), float(ally.max())
        pad = (y1 - y0) * 0.08 or 1.0
        y0, y1 = y0 - pad, y1 + pad
    else:
        y0, y1 = y_extent

    def X(v):
        if xlog:
            return m_left + (np.log10(v) - np.log10(x0)) / (np.log10(x1) - np.log10(x0)) * (plot_w - 1)
        return m_left + (v - x0) / (x1 - x0) * (plot_w - 1)

    def Y(v):
        return m_top + (1 - (v - y0) / (y1 - y0)) * (plot_h - 1)

    if xlog:
        xticks = np.geomspace(x0, x1, 8)
    else:
        xticks = nice_ticks(x0, x1, 6)
    for t in xticks:
        if x0 <= t <= x1 + 1e-12:
            xp = X(t)
            d.line([xp, m_top, xp, m_top + plot_h - 1], fill=(232, 232, 232), width=1)
            d.line([xp, m_top + plot_h - 1, xp, m_top + plot_h + 8], fill="black", width=2)
            d.text((xp, m_top + plot_h + 16), _fmt(t), font=f_tick, fill="black", anchor="ma")
    for t in nice_ticks(y0, y1, 6):
        yp = Y(t)
        d.line([m_left, yp, m_left + plot_w - 1, yp], fill=(232, 232, 232), width=1)
        d.line([m_left - 8, yp, m_left - 1, yp], fill="black", width=2)
        d.text((m_left - 14, yp), _fmt(t), font=f_tick, fill="black", anchor="rm")

    d.rectangle([m_left, m_top, m_left + plot_w - 1, m_top + plot_h - 1], outline="black")

    for h in (hlines or []):
        yp = Y(h["y"])
        d.line([m_left, yp, m_left + plot_w - 1, yp], fill=h.get("color", "gray"), width=2)
        if h.get("label"):
            d.text((m_left + 8, yp - 30), h["label"], font=f_legend, fill=h.get("color", "gray"), anchor="lm")

    for s in series:
        color = s.get("color", "black")
        pts = [(X(float(xv)), Y(float(yv))) for xv, yv in zip(s["x"], s["y"])]
        d.line(pts, fill=color, width=s.get("width", 4))

    ly = m_top + 14
    for s in series:
        color = s.get("color", "black")
        d.line([m_left + 14, ly, m_left + 44, ly], fill=color, width=4)
        d.text((m_left + 52, ly), s.get("label", ""), font=f_legend, fill="black", anchor="lm")
        ly += 38

    d.text((m_left + plot_w / 2, m_top + plot_h + 60), xlabel, font=f_axis, fill="black", anchor="ma")
    d.text((m_left - 80, m_top + plot_h / 2), ylabel, font=f_axis, fill="black", anchor="mm")

    img.save(path)


# --------------------------------------------------------------------------- #
# Datos y generación de cada imagen.
# --------------------------------------------------------------------------- #
LAM = 0.8e-6
FN = 0.25
POWER = 1.0e15
UM = 1.0e6


def _compute_slices():
    Ec_xz, ax_x, ax_z, _ = focus.focal_field_slice(
        plane="y", coord_m=0.0, lambda0_m=LAM, f_number=FN, power_w=POWER,
        n1=300, n2=300, span1_m=3.0e-6, span2_m=3.0e-6, ntheta=48, nphi=96,
    )
    I_xz = np.abs(Ec_xz[0]) ** 2 + np.abs(Ec_xz[1]) ** 2 + np.abs(Ec_xz[2]) ** 2
    Ec_yz, ax_z2, ax_y, _ = focus.focal_field_slice(
        plane="x", coord_m=0.0, lambda0_m=LAM, f_number=FN, power_w=POWER,
        n1=300, n2=300, span1_m=3.0e-6, span2_m=6.0e-6, ntheta=48, nphi=96,
    )
    I_yz = np.abs(Ec_yz[0]) ** 2 + np.abs(Ec_yz[1]) ** 2 + np.abs(Ec_yz[2]) ** 2
    return (I_xz, ax_x, ax_z), (I_yz, ax_z2, ax_y)


def image_focal_intensity(out: Path, slices) -> None:
    (I_xz, ax_x, ax_z), _ = slices
    logI = np.log10(I_xz + 1e-30)
    vmax = float(logI.max())
    draw_heatmap(
        out, logI,
        x_extent=(ax_x[0] * UM, ax_x[-1] * UM), y_extent=(ax_z[0] * UM, ax_z[-1] * UM),
        title="Campo focal |E|² (plano y = 0)",
        xlabel="x (µm)", ylabel="z (µm)", clabel="log₁₀|E|²",
        vmin=vmax - 6.0, vmax=vmax,
    )


def image_convergence(out: Path, slices) -> None:
    _, (I_yz, ax_z2, ax_y) = slices
    logI = np.log10(I_yz + 1e-30)
    vmax = float(logI.max())
    draw_heatmap(
        out, logI,
        x_extent=(ax_z2[0] * UM, ax_z2[-1] * UM), y_extent=(ax_y[0] * UM, ax_y[-1] * UM),
        title="Convergencia del haz (plano x = 0)",
        xlabel="z (µm)", ylabel="y (µm)", clabel="log₁₀|E|²",
        vmin=vmax - 6.0, vmax=vmax,
    )


def image_profile(out: Path, slices) -> None:
    (I_xz, ax_x, _), (I_yz, _, ax_y) = slices
    ex = np.sqrt(I_xz[I_xz.shape[0] // 2, :])        # |E(x, 0, 0)|
    ey = np.sqrt(I_yz[:, I_yz.shape[1] // 2])        # |E(0, y, 0)|
    draw_line_plot(
        out,
        [
            {"x": ax_x * UM, "y": ex, "label": "|E| vs x (transversal)", "color": (200, 30, 40)},
            {"x": ax_y * UM, "y": ey, "label": "|E| vs y (longitudinal)", "color": (30, 80, 200)},
        ],
        title="Perfil del campo en el foco",
        xlabel="posición (µm)", ylabel="|E| (V/m)",
    )


def image_kappa_sweep(out: Path) -> None:
    fns = np.concatenate(([0.25], np.geomspace(0.35, 4.0, 12)))
    res = sweep.kappa_eff_vs_f_number(
        fns, lambda0_m=1.0e-6, power_w=POWER, energy_gev=1.0, ntheta=128, nphi=256,
    )
    f = np.array([r["f_number"] for r in res])
    chi = np.array([r["chi_max_per_GeV"] for r in res])
    draw_line_plot(
        out,
        [{"x": f, "y": chi, "label": "χ_max por GeV", "color": (30, 80, 200)}],
        title="χ_max en función del f-number (Fig. 4)",
        xlabel="f-number  L/(2 R_max)", ylabel="χ_max por GeV",
        xlog=True,
        hlines=[{"y": 5.25, "color": (180, 30, 30), "label": "límite bi-dipolo κ = 5.25"}],
    )


def image_farfield(out: Path) -> None:
    th = np.linspace(0.0, np.pi / 2, 300)
    ph = np.linspace(0.0, 2 * np.pi, 96, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(TH) * np.cos(PH), -np.cos(TH), np.sin(TH) * np.sin(PH)], axis=-1)
    E = focus.bi_dipole_far_field(n)
    I = np.mean(np.sum(np.abs(E) ** 2, axis=-1), axis=1)  # promedio azimutal
    draw_line_plot(
        out,
        [{"x": th * 180.0 / np.pi, "y": I, "label": "|E_ff|² (promedio azimutal)", "color": (30, 80, 200)}],
        title="Distribución angular del campo lejano (bi-dipolo)",
        xlabel="θ desde el eje −y (grados)", ylabel="|E_ff|² (u. a.)",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Calculando cortes del campo focal...")
    slices = _compute_slices()
    jobs = [
        (OUT / "01_campo_focal_xz.png", image_focal_intensity, slices),
        (OUT / "02_convergencia_yz.png", image_convergence, slices),
        (OUT / "03_perfil_campo.png", image_profile, slices),
        (OUT / "04_chi_vs_fnumber.png", image_kappa_sweep, None),
        (OUT / "05_patron_angular.png", image_farfield, None),
    ]
    for path, fn, arg in jobs:
        print(f"Generando {path.name} ...")
        if arg is None:
            fn(path)
        else:
            fn(path, arg)
    print(f"\nListo. {len(jobs)} imágenes en {OUT.resolve()}")


if __name__ == "__main__":
    main()


