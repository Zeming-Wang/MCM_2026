from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def _project_root() -> Path:
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))).resolve()


def _ensure_import_path(project_root: Path) -> None:
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _format_float(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{float(x):.3f}"


def plot_trigger_curve(
    curve: pd.DataFrame,
    *,
    s0: float,
    metrics_text: str,
    out_path: Path,
    title: str,
) -> Path:
    if curve.empty:
        raise ValueError("Empty trigger curve.")

    plt.style.use("seaborn-v0_8-paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "#f2f0e9",
            "figure.facecolor": "#f2f0e9",
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=260)
    x = curve["S_mid"].to_numpy(dtype=float)
    y = curve["P_changed"].to_numpy(dtype=float)
    n = curve["N"].to_numpy(dtype=float)
    n_max = float(np.nanmax(n)) if np.isfinite(np.nanmax(n)) else 1.0
    sizes = 25.0 + 120.0 * np.sqrt(np.clip(n / (n_max if n_max > 0 else 1.0), 0.0, 1.0))

    ax.plot(x, y, color="#3b5b92", lw=2.4, alpha=0.9, zorder=2)
    ax.scatter(
        x,
        y,
        s=sizes,
        color="#3b5b92",
        edgecolor="white",
        linewidth=0.9,
        alpha=0.9,
        zorder=3,
    )

    ax.axvline(float(s0), color="#b04a3a", lw=2.0, alpha=0.85, zorder=1)
    ax.text(
        float(s0),
        1.02,
        "S0",
        ha="center",
        va="bottom",
        transform=ax.get_xaxis_transform(),
        fontsize=10.5,
        color="#b04a3a",
    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Sensitivity S = −log10(|Fan_Gap_Bottom2|)", fontsize=12)
    ax.set_ylabel("P(Changed=1)", fontsize=12)
    ax.set_title(title, fontsize=13.0, pad=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.28)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#1f1f1f",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d8d6ce", alpha=0.88),
    )

    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--season_type", type=str, default="all")
    p.add_argument("--quantile", type=float, default=0.8)
    p.add_argument("--bins", type=int, default=12)
    p.add_argument("--out_png", type=str, default="")
    p.add_argument("--out_curve_csv", type=str, default="")
    args = p.parse_args()

    project_root = _project_root()
    _ensure_import_path(project_root)

    from src.models.revote_decision_minimal_metrics import (
        find_latest_geometry_table,
        load_geometry_table,
        compute_minimal_metrics,
        build_binned_trigger_curve,
    )

    csv_path = Path(args.csv).resolve() if args.csv else find_latest_geometry_table(project_root)
    df = load_geometry_table(csv_path)

    season_type = None if str(args.season_type).strip().lower() in {"", "all", "none"} else str(args.season_type)
    m = compute_minimal_metrics(df, s_quantile=float(args.quantile), season_type=season_type)
    curve = build_binned_trigger_curve(df, bins=int(args.bins), season_type=season_type)

    curve_csv = (
        Path(args.out_curve_csv).resolve()
        if args.out_curve_csv
        else (project_root / "scripts" / "visualization" / "outputs" / "revote_decision_minimal_trigger_curve.csv")
    )
    curve_csv.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_csv, index=False, encoding="utf-8-sig")

    out_png = (
        Path(args.out_png).resolve()
        if args.out_png
        else (project_root / "charts" / "revote_decision_minimal_trigger_curve.png")
    )

    group_name = "all" if season_type is None else f"Season_Type={season_type}"
    metrics_text = (
        f"{group_name}\n"
        f"S0(q={_format_float(float(args.quantile))}) = {_format_float(m.s0)}\n"
        f"R_high = {_format_float(m.r_high)}\n"
        f"R_low  = {_format_float(m.r_low)}\n"
        f"G      = {_format_float(m.g)}\n"
        f"N={m.n_total} (high={m.n_high}, low={m.n_low})"
    )
    title = "Revote Trigger Curve (Minimal Decision)"
    if season_type is not None:
        title += f" — {season_type}"

    saved = plot_trigger_curve(
        curve,
        s0=m.s0,
        metrics_text=metrics_text,
        out_path=out_png,
        title=title,
    )
    print(f"Input: {csv_path}")
    print(f"Saved curve csv: {curve_csv}")
    print(f"Saved figure: {saved}")


if __name__ == "__main__":
    main()
