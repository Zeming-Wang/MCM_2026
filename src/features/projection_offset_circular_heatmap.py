from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist


def _load_projection_offset_pivot(project_root: Path) -> pd.DataFrame:
    summary_csv = (
        project_root
        / "scripts"
        / "visualization"
        / "outputs"
        / "projection_consistency_summary.csv"
    )
    df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype(int)
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype(int)
    df["ProjectionOffset_L1"] = pd.to_numeric(df["ProjectionOffset_L1"], errors="coerce")
    pivot = (
        df.pivot_table(
            index="Season",
            columns="Week",
            values="ProjectionOffset_L1",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    return pivot


def _cluster_row_order(pivot: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    x = pivot.copy()
    col_median = x.median(axis=0, skipna=True)
    for c in x.columns:
        fill_val = float(col_median[c]) if not np.isnan(col_median[c]) else 0.0
        x[c] = x[c].fillna(fill_val)

    d = pdist(x.to_numpy(dtype=float), metric="euclidean")
    z = linkage(d, method="average")
    info = dendrogram(z, labels=[str(i) for i in x.index.tolist()], no_plot=True)
    leaf_labels = [int(s) for s in info["ivl"]]
    order_idx = [x.index.tolist().index(s) for s in leaf_labels]
    return z, order_idx


def _unwrap_short(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size <= 1:
        return theta
    out = np.empty_like(theta)
    out[0] = theta[0]
    two_pi = 2.0 * np.pi
    for i in range(1, theta.size):
        t = theta[i]
        prev = out[i - 1]
        while t - prev > np.pi:
            t -= two_pi
        while t - prev < -np.pi:
            t += two_pi
        out[i] = t
    return out


def _plot_circular_cluster_heatmap(
    pivot: pd.DataFrame,
    out_path: Path,
) -> Path:
    pivot = pivot.copy()
    z, order_idx = _cluster_row_order(pivot)
    pivot = pivot.iloc[order_idx]

    seasons = pivot.index.to_list()
    weeks = pivot.columns.to_list()

    values = pivot.to_numpy(dtype=float)
    vmin_data = float(np.nanmin(values)) if np.isfinite(np.nanmin(values)) else 0.0
    vmax_data = float(np.nanmax(values)) if np.isfinite(np.nanmax(values)) else 1.0
    if vmax_data <= vmin_data:
        vmax_data = vmin_data + 1e-9

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "pval_like",
        ["#8c2d5e", "#f7fbff", "#7fcdbb"],
        N=256,
    )
    cmap = cmap.copy()
    cmap.set_bad(color="white")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(10.4, 10.4), dpi=260)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])

    ax.set_aspect("equal")
    ax.axis("off")

    n = len(seasons)
    m = len(weeks)

    start_angle = -np.pi / 2
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n + 1, endpoint=True) + start_angle

    r_center_outline = 0.18
    r_dend_inner = 0.24
    r_dend_outer = 0.64
    r_heat_inner = 0.70
    ring_gap = 0.006
    ring_thickness = (0.97 - r_heat_inner - (m - 1) * ring_gap) / max(m, 1)
    ring_thickness = float(np.clip(ring_thickness, 0.022, 0.052))

    set_link_color_palette(
        [
            "#7fcdbb",
            "#80b1d3",
            "#b3de69",
            "#bebada",
            "#fdb462",
            "#bc80bd",
            "#fb8072",
        ]
    )
    den = dendrogram(
        z,
        labels=[f"S{int(s)}" for s in seasons],
        no_plot=True,
        color_threshold=0.7 * float(np.max(z[:, 2])),
    )
    icoord = den["icoord"]
    dcoord = den["dcoord"]
    colors = den["color_list"]

    x_min = float(min(min(xs) for xs in icoord))
    x_max = float(max(max(xs) for xs in icoord))
    y_max = float(max(max(ys) for ys in dcoord)) if dcoord else 1.0
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= 0:
        y_max = 1.0

    for xs, ys, col in zip(icoord, dcoord, colors, strict=False):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        theta = start_angle + 2.0 * np.pi * (xs - x_min) / (x_max - x_min)
        theta = _unwrap_short(theta)
        r = r_dend_outer - (ys / y_max) * (r_dend_outer - r_dend_inner)
        xy = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.0, alpha=0.85, solid_capstyle="round")

    for j, w in enumerate(weeks):
        inner = r_heat_inner + j * (ring_thickness + ring_gap)
        outer = inner + ring_thickness
        for i, s in enumerate(seasons):
            v = pivot.loc[s, w]
            if pd.isna(v):
                face = (1.0, 1.0, 1.0, 1.0)
            else:
                frac = (float(v) - vmin_data) / (vmax_data - vmin_data)
                frac = float(np.clip(frac, 0.0, 1.0))
                frac = float(np.power(frac, 0.45))
                face = cmap(norm(frac))

            t0 = float(theta_edges[i])
            t1 = float(theta_edges[i + 1])
            deg0 = float(np.degrees(t0))
            deg1 = float(np.degrees(t1))
            patch = Wedge(
                (0.0, 0.0),
                r=outer,
                theta1=deg0,
                theta2=deg1,
                width=ring_thickness,
                facecolor=face,
                edgecolor="white",
                linewidth=0.38,
            )
            ax.add_patch(patch)

    r_text = r_heat_inner + m * (ring_thickness + ring_gap) + 0.04
    for i, s in enumerate(seasons):
        theta = float((theta_edges[i] + theta_edges[i + 1]) / 2.0)
        x = r_text * np.cos(theta)
        y = r_text * np.sin(theta)
        rot = float(np.degrees(theta) - 90.0)
        if 90.0 < (rot % 360.0) < 270.0:
            rot += 180.0
            ha = "right"
        else:
            ha = "left"
        ax.text(
            x,
            y,
            f"S{int(s)}",
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=5.6,
            color="#222222",
        )

    ax.add_patch(
        Circle(
            (0.0, 0.0),
            radius=r_center_outline,
            facecolor="none",
            edgecolor="#B5B5B5",
            linewidth=1.0,
            alpha=0.65,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            radius=r_dend_outer,
            facecolor="none",
            edgecolor="#D0D0D0",
            linewidth=0.9,
            alpha=0.55,
        )
    )

    cax = fig.add_axes([0.47, 0.43, 0.022, 0.16])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.set_title("pval", fontsize=11, weight="bold", pad=6)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_linewidth(0.8)

    legend_text = "\n".join(
        [
            "Weighted mode",
            "Simple mode",
            "Inverse variance weighted",
            "Weighted median",
            "MR Egger",
        ]
    )
    fig.text(
        0.73,
        0.70,
        legend_text,
        rotation=-35,
        ha="left",
        va="top",
        fontsize=10,
        color="#333333",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    project_root = Path(
        os.environ.get("MCM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    ).resolve()

    pivot = _load_projection_offset_pivot(project_root)
    out_path = project_root / "charts" / "projection_offset_summary_circular_heatmap_replica.png"
    _plot_circular_cluster_heatmap(pivot, out_path=out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

