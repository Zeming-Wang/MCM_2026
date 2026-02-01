from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def _project_root() -> Path:
    return Path(
        os.environ.get("MCM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    ).resolve()


def _load_geometry_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")
    df["Fan_Gap_Bottom2"] = pd.to_numeric(df["Fan_Gap_Bottom2"], errors="coerce")
    df = df.dropna(subset=["Season", "Week", "Fan_Gap_Bottom2"]).copy()
    df["Season"] = df["Season"].astype(int)
    df["Week"] = df["Week"].astype(int)
    df["Fan_Gap_Bottom2"] = df["Fan_Gap_Bottom2"].astype(float)
    return df


def _compute_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    x = np.abs(df["Fan_Gap_Bottom2"].to_numpy(dtype=float))
    eps = 1e-6
    df = df.copy()
    df["Sensitivity"] = -np.log10(np.clip(x, eps, None))
    df["Sensitivity"] = df["Sensitivity"].astype(float)
    return df


def _kde_on_grid(values: np.ndarray, grid: np.ndarray, bw_adjust: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(grid, dtype=float)
    if values.size == 1:
        sigma = max(1e-3, 0.035 * float(grid.max() - grid.min()))
        return np.exp(-0.5 * ((grid - float(values[0])) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    if float(np.nanstd(values)) <= 1e-10:
        sigma = max(1e-3, 0.03 * float(grid.max() - grid.min()))
        mu = float(np.nanmean(values))
        return np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    try:
        kde = gaussian_kde(values, bw_method="scott")
        kde.set_bandwidth(kde.factor * float(bw_adjust))
        y = kde(grid)
        y = np.maximum(y, 0.0)
        return y
    except Exception:
        sigma = max(1e-3, 0.05 * float(grid.max() - grid.min()))
        mu = float(np.nanmean(values))
        return np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def plot_ridgeline_sensitivity(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str | None = None,
    bw_adjust: float = 0.85,
) -> Path:
    df = _compute_sensitivity(df)
    seasons = sorted(df["Season"].unique().tolist())
    if not seasons:
        raise ValueError("No seasons found in geometry table.")

    all_vals = df["Sensitivity"].to_numpy(dtype=float)
    x_min = float(np.nanpercentile(all_vals, 1))
    x_max = float(np.nanpercentile(all_vals, 99))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min = float(np.nanmin(all_vals)) if np.isfinite(np.nanmin(all_vals)) else 0.0
        x_max = float(np.nanmax(all_vals)) if np.isfinite(np.nanmax(all_vals)) else x_min + 1.0
        if x_max <= x_min:
            x_max = x_min + 1.0
    pad = 0.12 * (x_max - x_min)
    x_min -= pad
    x_max += pad
    grid = np.linspace(x_min, x_max, 560, dtype=float)

    plt.style.use("seaborn-v0_8-paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "#f2f0e9",
            "figure.facecolor": "#f2f0e9",
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 7.0), dpi=260)

    cmap = plt.get_cmap("Spectral")
    colors = [cmap(v) for v in np.linspace(0.04, 0.96, max(len(seasons), 2))]
    colors = colors[: len(seasons)]

    ridge_height = 0.86
    y_positions = np.arange(len(seasons), dtype=float)[::-1]
    baseline_color = "#d0cdc4"

    for i, season in enumerate(seasons):
        sub = df[df["Season"] == int(season)]
        vals = sub["Sensitivity"].to_numpy(dtype=float)
        dens = _kde_on_grid(vals, grid, bw_adjust=bw_adjust)
        peak = float(np.nanmax(dens)) if dens.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            continue
        dens = dens / peak * ridge_height

        y0 = float(y_positions[i])
        ax.hlines(y0, x_min, x_max, color=baseline_color, lw=1.05, alpha=0.9, zorder=1)
        ax.fill_between(
            grid,
            y0,
            y0 + dens,
            color=colors[i],
            alpha=0.95,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(grid, y0 + dens, color="white", lw=1.25, alpha=0.98, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Season {s}" for s in seasons], fontsize=11)
    ax.tick_params(axis="y", length=0, pad=16, colors="#1f1f1f")
    ax.tick_params(axis="x", colors="#2a2a2a")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.65, float(len(seasons) - 1) + 0.95)
    ax.set_xlabel("Mechanism Sensitivity  (−log10(Fan_Gap_Bottom2))", fontsize=12, color="#1f1f1f")
    if title:
        ax.set_title(title, fontsize=13.5, pad=14, color="#1f1f1f")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=1.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    project_root = _project_root()
    csv_path = (
        project_root
        / "scripts"
        / "visualization"
        / "outputs"
        / "revote_compare_run_20260201_162034"
        / "bottom2_geometry_table.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    out_path = project_root / "charts" / "ridgeline_mechanism_sensitivity_scheme_b.png"
    df = _load_geometry_table(csv_path)
    saved = plot_ridgeline_sensitivity(
        df,
        out_path,
        title="Mechanism Sensitivity Distribution by Season",
    )
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
