
#!/usr/bin/env python3
"""Compact single-column academic motivation figure for DriftTraj.

The near-square 2+1 panel layout contrasts:

1. Standard WTA, where distinct query conditions can decode to redundant
   hypotheses and only the winner receives direct regression supervision.
2. Iterative diffusion and ODE-based flow-matching inference.
3. DriftTraj, whose GMP-conditioned parallel queries are shaped by a
   training-only, multi-scale MDF and remain one-pass at inference.

All geometry is generated with Matplotlib; no raster background is embedded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


DPI = 300
BLUE = "#58A6B8"
ORANGE = "#D98C6A"
BACKGROUND = "#FCFBF9"
BLACK = "#111111"
GRAY = "#707780"
LIGHT_GRAY = "#A8ADB3"
CURRENT = "#75808C"
PANEL_Y_MAX = 5.25

# Three separated target trajectory modes shared by every panel.
TARGET_MODES = (
    (8.18, 4.18, 0.82, 0.78, -14.0, 0.7),
    (7.96, 2.62, 1.02, 0.62, 10.0, 1.9),
    (8.12, 1.02, 0.86, 0.80, -4.0, 3.1),
)

# Generic mode-specific conditions used in the WTA panel.
WTA_CONDITIONS = (
    (2.15, 4.10, 0.86, 0.61, -14.0, 1.1),
    (2.42, 2.55, 0.72, 0.78, 12.0, 2.5),
    (2.12, 1.08, 0.90, 0.62, 4.0, 4.4),
)

# Conventional isotropic Gaussian prior used by diffusion / flow models.
ISOTROPIC_PRIOR = ((2.45, 2.55, 1.10, 1.48, -8.0, 1.7),)

# Compact multimodal GMP conditions. Their centers are intentionally closer
# than the data modes, highlighting a structured continuous prior rather than
# output anchors or a transported source distribution.
COMPACT_GMP = (
    (1.92, 1.55, 0.83, 0.58, -12.0, 1.0),
    (2.42, 3.45, 0.70, 0.72, 11.0, 2.7),
    (3.30, 2.34, 0.78, 0.56, -4.0, 4.5),
)


def preferred_times_font() -> str:
    """Prefer Times New Roman, then metrically compatible serif families."""
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    for family in ("Times New Roman", "Nimbus Roman", "STIXGeneral"):
        if family in installed:
            return family
    return "serif"


def configure_style() -> None:
    family = preferred_times_font()
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "mathtext.bfit": f"{family}:bold:italic",
            "mathtext.cal": family,
            "mathtext.sf": family,
            "mathtext.tt": family,
            "mathtext.fallback": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    mean: tuple[float, float],
    sigma_x: float,
    sigma_y: float,
    angle_deg: float = 0.0,
) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx, dy = xx - mean[0], yy - mean[1]
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    return np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))


def organic_mode(
    xx: np.ndarray,
    yy: np.ndarray,
    mean: tuple[float, float],
    sigma_x: float,
    sigma_y: float,
    *,
    phase: float,
    angle_deg: float = 0.0,
    roughness: float = 0.18,
) -> np.ndarray:
    """A slightly irregular Gaussian-like density, matching the reference."""
    base = gaussian_2d(xx, yy, mean, sigma_x, sigma_y, angle_deg)
    dx = (xx - mean[0]) / sigma_x
    dy = (yy - mean[1]) / sigma_y
    radius = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    taper = 1.0 - np.exp(-((radius / 0.95) ** 2))
    bend = (
        0.34 * np.sin(3.0 * theta + phase)
        + 0.14 * np.sin(5.0 * theta - 1.5 * phase)
        + 0.10 * np.cos(2.0 * theta + 0.4 * phase)
    )
    broad = 0.24 * np.sin(0.82 * xx + 0.55 * yy + phase)
    value = base * np.exp(roughness * (taper * bend + broad))
    return value / value.max()


def setup_axis(
    ax: plt.Axes,
    title: str,
    *,
    title_size: float,
) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, PANEL_Y_MAX)
    ax.set_aspect(1.0, adjustable="box")
    ax.set_axis_off()
    ax.add_patch(
        Rectangle(
            (0, 0),
            10,
            PANEL_Y_MAX,
            facecolor=BACKGROUND,
            edgecolor="none",
            zorder=-20,
        )
    )
    ax.text(
        0.0,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=title_size,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
    )


def draw_contours(
    ax: plt.Axes,
    *,
    blue_modes: tuple[tuple[float, float, float, float, float, float], ...],
    orange_modes: tuple[tuple[float, float, float, float, float, float], ...],
) -> None:
    """Draw irregular pastel density contours.

    Each mode is (mean_x, mean_y, sigma_x, sigma_y, angle_deg, phase).
    """
    grid_x = np.linspace(0, 10, 520)
    grid_y = np.linspace(0, PANEL_Y_MAX, 300)
    xx, yy = np.meshgrid(grid_x, grid_y)

    def combine(
        modes: tuple[tuple[float, float, float, float, float, float], ...]
    ) -> np.ndarray:
        fields = [
            organic_mode(
                xx,
                yy,
                (mx, my),
                sx,
                sy,
                angle_deg=angle,
                phase=phase,
            )
            for mx, my, sx, sy, angle, phase in modes
        ]
        density = np.maximum.reduce(fields)
        return density / density.max()

    density_orange = combine(orange_modes)
    density_blue = combine(blue_modes)
    levels = [0.055, 0.16, 0.29, 0.43, 0.59, 0.77, 1.01]
    orange_alpha = [0.032, 0.062, 0.095, 0.128, 0.162, 0.195]
    blue_alpha = [0.015, 0.030, 0.046, 0.063, 0.080, 0.095]

    ax.contourf(
        xx,
        yy,
        density_orange,
        levels=levels,
        colors=[mpl.colors.to_rgba(ORANGE, alpha) for alpha in orange_alpha],
        antialiased=True,
        zorder=1,
    )
    ax.contour(
        xx,
        yy,
        density_orange,
        levels=levels[:-1],
        colors=[ORANGE],
        linewidths=0.48,
        alpha=0.27,
        zorder=2,
    )
    ax.contourf(
        xx,
        yy,
        density_blue,
        levels=levels,
        colors=[mpl.colors.to_rgba(BLUE, alpha) for alpha in blue_alpha],
        antialiased=True,
        zorder=3,
    )
    ax.contour(
        xx,
        yy,
        density_blue,
        levels=levels[:-1],
        colors=[BLUE],
        linewidths=0.48,
        alpha=0.19,
        zorder=4,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    linewidth: float = 1.25,
    mutation_scale: float = 10.0,
    dashed: bool = False,
    color: str = BLACK,
    alpha: float = 1.0,
    zorder: float = 15,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=(0, (3.0, 2.1)) if dashed else "solid",
            color=color,
            alpha=alpha,
            shrinkA=0,
            shrinkB=0,
            capstyle="round",
            joinstyle="round",
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def catmull_rom_chain(
    control_points: tuple[tuple[float, float], ...],
    samples_per_segment: int = 32,
) -> np.ndarray:
    """Return a smooth curve that passes through each control point."""
    points = np.asarray(control_points, dtype=float)
    padded = np.vstack([points[0], points, points[-1]])
    pieces: list[np.ndarray] = []
    for index in range(len(points) - 1):
        p0, p1, p2, p3 = padded[index : index + 4]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
        segment = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
        )
        pieces.append(segment)
    return np.vstack([*pieces, points[-1]])


def draw_path(
    ax: plt.Axes,
    control_points: tuple[tuple[float, float], ...],
    *,
    color: str,
    dashed: bool,
    marker: str,
    marker_count: int,
    linewidth: float,
    arrow_fractions: tuple[float, ...],
) -> np.ndarray:
    curve = catmull_rom_chain(control_points)
    ax.plot(
        curve[:, 0],
        curve[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=(0, (2.2, 2.0)) if dashed else "solid",
        alpha=0.88,
        zorder=10,
    )
    indices = np.linspace(9, len(curve) - 10, marker_count, dtype=int)
    ax.scatter(
        curve[indices, 0],
        curve[indices, 1],
        s=15,
        marker=marker,
        facecolors="white",
        edgecolors=color,
        linewidths=0.70,
        zorder=12,
    )
    for fraction in arrow_fractions:
        center = int(fraction * (len(curve) - 1))
        arrow(
            ax,
            tuple(curve[max(0, center - 4)]),
            tuple(curve[min(len(curve) - 1, center + 4)]),
            linewidth=linewidth,
            mutation_scale=7.0,
            color=color,
            zorder=13,
        )
    return curve


def wta_panel(ax: plt.Axes) -> None:
    setup_axis(
        ax,
        "(a) Standard WTA: Mode Aliasing",
        title_size=6.5,
    )
    ax.set_anchor("S")
    draw_contours(ax, blue_modes=TARGET_MODES, orange_modes=WTA_CONDITIONS)

    queries = np.array([(2.30, 4.04), (2.08, 2.58), (2.42, 1.17)])
    predictions = np.array(
        [
            (6.50, 2.72),
            (6.68, 2.56),
            (6.64, 2.84),
            (6.84, 2.66),
            (6.79, 2.93),
            (6.94, 2.80),
        ]
    )
    winner = tuple(predictions[-1])
    gt = (7.95, 2.83)

    for index, (query, prediction) in enumerate(zip(queries, predictions[:3])):
        rad = (0.18, 0.0, -0.18)[index]
        arrow(
            ax,
            tuple(query),
            tuple(prediction),
            linewidth=0.85,
            mutation_scale=7.4,
            color=GRAY,
            alpha=0.78,
            zorder=8,
            connectionstyle=f"arc3,rad={rad}",
        )

    ax.scatter(
        queries[:, 0],
        queries[:, 1],
        s=15,
        marker="D",
        color=ORANGE,
        edgecolors="white",
        linewidths=0.65,
        zorder=18,
    )
    for index, query in enumerate(queries, start=1):
        ax.text(
            query[0] - 0.27,
            query[1] + 0.25,
            rf"$r_{index}$",
            fontsize=5.8,
            ha="right",
            va="bottom",
            color=BLACK,
            zorder=20,
        )

    ax.scatter(
        predictions[:-1, 0],
        predictions[:-1, 1],
        s=15,
        marker="o",
        facecolors="white",
        edgecolors=GRAY,
        linewidths=0.90,
        zorder=18,
    )
    ax.scatter(
        winner[0],
        winner[1],
        s=15,
        marker="o",
        color=BLACK,
        edgecolors="white",
        linewidths=0.55,
        zorder=21,
    )
    ax.scatter(
        gt[0],
        gt[1],
        s=24,
        marker="*",
        color=BLUE,
        edgecolors="white",
        linewidths=0.65,
        zorder=22,
    )
    arrow(
        ax,
        winner,
        (7.72, 2.83),
        linewidth=1.20,
        mutation_scale=9.2,
        dashed=True,
        color=BLUE,
        zorder=17,
    )
    ax.text(
        4.55,
        4.45,
        r"shared decoder  $f_{\theta}$",
        fontsize=5.2,
        fontweight="bold",
        ha="center",
        va="center",
        color=BLACK,
        zorder=20,
    )
    ax.text(
        2.15,
        0.18,
        "query / prior conditions",
        fontsize=4.7,
        fontstyle="italic",
        ha="center",
        va="bottom",
        color=ORANGE,
        zorder=20,
    )
    ax.text(
        8.22,
        0.18,
        "target modes",
        fontsize=4.7,
        fontstyle="italic",
        ha="center",
        va="bottom",
        color=BLUE,
        zorder=20,
    )
    ax.text(
        gt[0] + 0.17,
        gt[1] + 0.09,
        r"$\mathbf{Y}_{\rm GT}$",
        fontsize=6.6,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=BLACK,
        zorder=23,
    )


def iterative_panel(ax: plt.Axes) -> None:
    setup_axis(
        ax,
        "(b) Iterative Generative Models",
        title_size=6.4,
    )
    ax.set_anchor("S")
    draw_contours(ax, blue_modes=TARGET_MODES, orange_modes=ISOTROPIC_PRIOR)

    diffusion_start = (2.22, 3.24)
    flow_start = (2.28, 1.55)
    gt = (7.90, 2.79)

    diffusion_curve = draw_path(
        ax,
        (
            diffusion_start,
            (2.78, 4.12),
            (3.60, 3.55),
            (4.36, 4.20),
            (5.05, 3.15),
            (5.82, 3.83),
            (6.62, 2.92),
            gt,
        ),
        color=GRAY,
        dashed=True,
        marker="o",
        marker_count=7,
        linewidth=1.05,
        arrow_fractions=(0.22, 0.48, 0.73),
    )
    flow_curve = draw_path(
        ax,
        (
            flow_start,
            (3.75, 1.78),
            (5.20, 2.14),
            (6.57, 2.46),
            gt,
        ),
        color=BLACK,
        dashed=False,
        marker="s",
        marker_count=5,
        linewidth=1.18,
        arrow_fractions=(0.31, 0.65),
    )

    cloud = np.array(
        [
            (4.75, 2.92),
            (5.06, 2.50),
            (5.22, 2.96),
            (5.48, 2.68),
            (5.76, 3.15),
            (5.98, 2.72),
            (6.30, 2.86),
        ]
    )
    ax.scatter(
        cloud[:, 0],
        cloud[:, 1],
        s=9,
        marker="o",
        color=CURRENT,
        edgecolors="white",
        linewidths=0.30,
        alpha=0.73,
        zorder=9,
    )

    ax.scatter(
        diffusion_start[0],
        diffusion_start[1],
        s=15,
        marker="D",
        color=ORANGE,
        edgecolors="white",
        linewidths=0.65,
        zorder=19,
    )
    ax.scatter(
        flow_start[0],
        flow_start[1],
        s=15,
        marker="s",
        color=ORANGE,
        edgecolors="white",
        linewidths=0.65,
        zorder=19,
    )
    ax.scatter(
        gt[0],
        gt[1],
        s=24,
        marker="*",
        color=BLUE,
        edgecolors="white",
        linewidths=0.65,
        zorder=22,
    )

    ax.text(
        4.22,
        4.58,
        r"Diffusion: $K$ denoising steps",
        fontsize=5.0,
        fontweight="bold",
        ha="center",
        va="center",
        color=GRAY,
        zorder=20,
    )
    ax.text(
        4.55,
        0.96,
        r"ODE-based FM: $N$ field evaluations",
        fontsize=4.9,
        fontweight="bold",
        ha="center",
        va="center",
        color=BLACK,
        zorder=20,
    )
    ax.text(
        4.18,
        2.62,
        r"$\mathbf{x}_t\sim r_t$",
        fontsize=5.6,
        ha="center",
        va="center",
        color=CURRENT,
        zorder=20,
    )
    ax.text(
        2.18,
        0.18,
        r"isotropic prior  $\mathcal{N}(\mathbf{0},\mathbf{I})$",
        fontsize=4.6,
        fontstyle="italic",
        ha="center",
        va="bottom",
        color=ORANGE,
        zorder=20,
    )
    ax.text(
        8.24,
        0.18,
        "target modes",
        fontsize=4.7,
        fontstyle="italic",
        ha="center",
        va="bottom",
        color=BLUE,
        zorder=20,
    )
    ax.text(
        gt[0] + 0.16,
        gt[1] + 0.10,
        r"$\mathbf{Y}_{\rm GT}$",
        fontsize=6.6,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=BLACK,
        zorder=23,
    )

    # Keep references alive for static analyzers and make it explicit that both
    # paths terminate at the same target rather than representing two datasets.
    _ = diffusion_curve, flow_curve


def drift_panel(ax: plt.Axes) -> None:
    setup_axis(
        ax,
        "(c) DriftTraj: GMP + Training-only MDF",
        title_size=8.6,
    )
    ax.set_aspect(0.8, adjustable="box")
    draw_contours(ax, blue_modes=TARGET_MODES, orange_modes=COMPACT_GMP)

    winner = (5.28, 2.66)
    gt = (7.78, 2.42)
    non_winners = np.array(
        [
            (3.92, 1.73),
            (4.17, 2.10),
            (4.24, 2.63),
            (4.47, 3.03),
            (4.63, 1.89),
            (4.70, 2.43),
            (4.18, 3.31),
        ]
    )

    for radius, alpha in ((0.40, 0.34), (0.76, 0.27), (1.12, 0.21)):
        ax.add_patch(
            Circle(
                winner,
                radius,
                fill=False,
                edgecolor=GRAY,
                linewidth=0.68,
                linestyle=(0, (2.2, 2.2)),
                alpha=alpha,
                zorder=7,
            )
        )
    ax.text(
        4.55,
        3.92,
        "multi-scale radius",
        fontsize=6.5,
        ha="center",
        va="bottom",
        color=GRAY,
        zorder=20,
    )

    ax.scatter(
        non_winners[:, 0],
        non_winners[:, 1],
        s=28,
        marker="x",
        color=ORANGE,
        linewidths=1.25,
        zorder=15,
    )
    for index in (2, 3, 5):
        point = non_winners[index]
        ax.plot(
            [point[0], winner[0]],
            [point[1], winner[1]],
            color=ORANGE,
            linewidth=0.58,
            linestyle=(0, (1.8, 2.1)),
            alpha=0.35,
            zorder=8,
        )

    ax.scatter(
        winner[0],
        winner[1],
        s=45,
        marker="o",
        color=BLACK,
        edgecolors="white",
        linewidths=0.55,
        zorder=22,
    )
    ax.scatter(
        gt[0],
        gt[1],
        s=88,
        marker="*",
        color=BLUE,
        edgecolors="white",
        linewidths=0.65,
        zorder=22,
    )

    attraction_end = (7.40, 2.46)
    repulsion_end = (6.34, 3.48)
    resultant_end = (7.18, 2.96)
    arrow(
        ax,
        winner,
        attraction_end,
        linewidth=1.35,
        mutation_scale=10.5,
        dashed=True,
        color=BLUE,
        zorder=17,
    )
    arrow(
        ax,
        winner,
        repulsion_end,
        linewidth=1.35,
        mutation_scale=10.5,
        dashed=True,
        color=ORANGE,
        zorder=17,
    )
    arrow(
        ax,
        winner,
        resultant_end,
        linewidth=1.95,
        mutation_scale=13.0,
        color=BLACK,
        zorder=18,
    )

    ax.text(
        gt[0] + 0.16,
        gt[1] + 0.08,
        r"$\mathbf{Y}_{\rm GT}$",
        fontsize=7.6,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=BLACK,
        zorder=23,
    )
    ax.text(
        winner[0] - 0.10,
        winner[1] + 0.20,
        r"winner  $\hat{\mathbf{Y}}_{k^{*}}$",
        fontsize=7.0,
        fontweight="bold",
        ha="right",
        va="bottom",
        color=BLACK,
        zorder=23,
    )
    ax.text(
        4.05,
        1.40,
        r"non-winners  $\hat{\mathbf{Y}}_{k\ne k^{*}}$",
        fontsize=6.3,
        ha="center",
        va="top",
        color=ORANGE,
        zorder=20,
    )
    ax.text(
        6.74,
        2.05,
        r"$\mathbf{F}_{\rm att}^{\rm GT}$",
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        va="top",
        color=BLUE,
        zorder=23,
    )
    ax.text(
        6.33,
        3.63,
        r"$\mathbf{F}_{\rm rep}^{\rm non\!-\!win}$",
        fontsize=6.7,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=ORANGE,
        zorder=23,
    )
    ax.text(
        6.94,
        3.10,
        r"$\mathbf{F}_{\rm MDF}$",
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=BLACK,
        zorder=23,
    )
    ax.text(
        2.38,
        0.18,
        "GMP query conditions",
        fontsize=6.2,
        fontstyle="italic",
        color=ORANGE,
        ha="center",
        va="bottom",
        zorder=20,
    )
    ax.text(
        8.24,
        0.18,
        "target trajectory modes",
        fontsize=6.2,
        fontstyle="italic",
        color=BLUE,
        ha="center",
        va="bottom",
        zorder=20,
    )


def build_figure() -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(3.42, 3.55), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.018,
        right=0.982,
        top=0.958,
        bottom=0.025,
        width_ratios=(1.0, 1.0),
        height_ratios=(1.6, 2.0),
        wspace=0.055,
        hspace=0.13,
    )
    wta_panel(fig.add_subplot(grid[0, 0]))
    iterative_panel(fig.add_subplot(grid[0, 1]))
    drift_ax = fig.add_subplot(grid[1, :])
    drift_panel(drift_ax)
    position = drift_ax.get_position()
    scale = 0.85
    drift_ax.set_position(
        [
            position.x0 + position.width * (1.0 - scale) / 2.0,
            position.y1 - position.height * scale,
            position.width * scale,
            position.height * scale,
        ]
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--png",
        type=Path,
        default=script_dir
        / "output"
        / "image"
        / "drifttraj_motivation_single_column.png",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=script_dir
        / "output"
        / "pdf"
        / "drifttraj_motivation_single_column.pdf",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure()

    png_temp = args.png.with_name(f".{args.png.stem}.tmp.png")
    pdf_temp = args.pdf.with_name(f".{args.pdf.stem}.tmp.pdf")
    figure.savefig(
        png_temp,
        dpi=args.dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    figure.savefig(
        pdf_temp,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)
    png_temp.replace(args.png)
    pdf_temp.replace(args.pdf)
    print(f"font={preferred_times_font()}")
    print(f"wrote {args.png}")
    print(f"wrote {args.pdf}")


if __name__ == "__main__":
    main()
