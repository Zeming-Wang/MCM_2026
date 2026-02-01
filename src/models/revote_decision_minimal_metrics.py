from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(
        os.environ.get("MCM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    ).resolve()


def find_latest_geometry_table(root: Path) -> Path:
    base = root / "scripts" / "visualization" / "outputs"
    candidates = list(base.glob("revote_compare_run_*/*bottom2_geometry_table.csv"))
    candidates += list(base.glob("revote_compare_run_*/bottom2_geometry_table.csv"))
    if not candidates:
        candidates = list(base.rglob("bottom2_geometry_table.csv"))
    if not candidates:
        raise FileNotFoundError(f"bottom2_geometry_table.csv not found under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_geometry_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["Season"] = pd.to_numeric(df.get("Season"), errors="coerce").astype("Int64")
    df["Week"] = pd.to_numeric(df.get("Week"), errors="coerce").astype("Int64")
    df["Season_Type"] = df.get("Season_Type", "unknown").astype(str)
    df["Changed"] = pd.to_numeric(df.get("Changed"), errors="coerce")
    df["Fan_Gap_Bottom2"] = pd.to_numeric(df.get("Fan_Gap_Bottom2"), errors="coerce")
    df["Judge_Gap_Bottom2"] = pd.to_numeric(df.get("Judge_Gap_Bottom2"), errors="coerce")
    df = df.dropna(subset=["Season", "Week", "Changed", "Fan_Gap_Bottom2", "Judge_Gap_Bottom2"]).copy()
    df["Season"] = df["Season"].astype(int)
    df["Week"] = df["Week"].astype(int)
    df["Changed"] = (df["Changed"].astype(float) > 0).astype(int)
    df["Fan_Gap_Bottom2"] = df["Fan_Gap_Bottom2"].astype(float)
    df["Judge_Gap_Bottom2"] = df["Judge_Gap_Bottom2"].astype(float)
    return df


def add_sensitivity(
    df: pd.DataFrame,
    *,
    eps: float = 1e-6,
    fan_gap_col: str = "Fan_Gap_Bottom2",
) -> pd.DataFrame:
    df = df.copy()
    fan_gap = np.abs(df[fan_gap_col].to_numpy(dtype=float))
    df["Sensitivity"] = -np.log10(np.clip(fan_gap, eps, None))
    df["Sensitivity"] = df["Sensitivity"].astype(float)
    return df


@dataclass(frozen=True)
class MinimalDecisionMetrics:
    s0: float
    r_high: float
    r_low: float
    g: float
    n_total: int
    n_high: int
    n_low: int
    n_changed_total: int
    n_changed_high: int
    n_changed_low: int


def compute_minimal_metrics(
    df: pd.DataFrame,
    *,
    s_quantile: float = 0.8,
    season_type: str | None = None,
) -> MinimalDecisionMetrics:
    df = add_sensitivity(df)
    if season_type is not None:
        df = df[df["Season_Type"].astype(str) == str(season_type)].copy()
    if df.empty:
        raise ValueError("No rows available after filtering.")

    s0 = float(np.nanquantile(df["Sensitivity"].to_numpy(dtype=float), float(s_quantile)))
    high = df[df["Sensitivity"] >= s0].copy()
    low = df[df["Sensitivity"] < s0].copy()

    def _rate(x: pd.DataFrame) -> float:
        if x.empty:
            return float("nan")
        return float(x["Changed"].mean())

    r_high = _rate(high)
    r_low = _rate(low)

    abs_j = np.abs(high["Judge_Gap_Bottom2"].to_numpy(dtype=float))
    denom = float(np.nanmedian(abs_j)) if abs_j.size else float("nan")
    abs_j_changed = np.abs(
        high.loc[high["Changed"] == 1, "Judge_Gap_Bottom2"].to_numpy(dtype=float)
    )
    num = float(np.nanmedian(abs_j_changed)) if abs_j_changed.size else float("nan")
    if not np.isfinite(denom) or denom <= 0.0 or not np.isfinite(num):
        g = float("nan")
    else:
        g = float(num / denom)

    return MinimalDecisionMetrics(
        s0=s0,
        r_high=r_high,
        r_low=r_low,
        g=g,
        n_total=int(len(df)),
        n_high=int(len(high)),
        n_low=int(len(low)),
        n_changed_total=int(df["Changed"].sum()),
        n_changed_high=int(high["Changed"].sum()) if not high.empty else 0,
        n_changed_low=int(low["Changed"].sum()) if not low.empty else 0,
    )


def build_binned_trigger_curve(
    df: pd.DataFrame,
    *,
    bins: int = 12,
    season_type: str | None = None,
) -> pd.DataFrame:
    df = add_sensitivity(df)
    if season_type is not None:
        df = df[df["Season_Type"].astype(str) == str(season_type)].copy()
    if df.empty:
        return pd.DataFrame(columns=["S_bin_left", "S_bin_right", "S_mid", "P_changed", "N"])

    s = df["Sensitivity"].to_numpy(dtype=float)
    s_min = float(np.nanmin(s))
    s_max = float(np.nanmax(s))
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
        s_min = 0.0
        s_max = s_min + 1.0
    edges = np.linspace(s_min, s_max, int(bins) + 1, dtype=float)
    idx = np.digitize(s, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    df = df.copy()
    df["_bin"] = idx
    g = df.groupby("_bin", dropna=False)
    out = g.agg(
        P_changed=("Changed", "mean"),
        N=("Changed", "size"),
        S_median=("Sensitivity", "median"),
    ).reset_index(drop=True)
    out["S_bin_left"] = edges[: len(out)]
    out["S_bin_right"] = edges[1 : len(out) + 1]
    out["S_mid"] = (out["S_bin_left"] + out["S_bin_right"]) / 2.0
    out = out[["S_bin_left", "S_bin_right", "S_mid", "S_median", "P_changed", "N"]]
    out["P_changed"] = out["P_changed"].astype(float)
    out["N"] = out["N"].astype(int)
    return out


def metrics_to_frame(m: MinimalDecisionMetrics, *, label: str, s_quantile: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Group": label,
                "S0_quantile": float(s_quantile),
                "S0_value": m.s0,
                "R_high": m.r_high,
                "R_low": m.r_low,
                "G": m.g,
                "N_total": m.n_total,
                "N_high": m.n_high,
                "N_low": m.n_low,
                "N_changed_total": m.n_changed_total,
                "N_changed_high": m.n_changed_high,
                "N_changed_low": m.n_changed_low,
            }
        ]
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--quantile", type=float, default=0.8)
    p.add_argument("--out_csv", type=str, default="")
    args = p.parse_args()

    root = project_root()
    csv_path = Path(args.csv).resolve() if args.csv else find_latest_geometry_table(root)
    df = load_geometry_table(csv_path)

    all_m = compute_minimal_metrics(df, s_quantile=float(args.quantile))
    frames = [metrics_to_frame(all_m, label="all", s_quantile=float(args.quantile))]

    for st in sorted(df["Season_Type"].astype(str).unique().tolist()):
        try:
            frames.append(
                metrics_to_frame(
                    compute_minimal_metrics(df, s_quantile=float(args.quantile), season_type=st),
                    label=f"Season_Type={st}",
                    s_quantile=float(args.quantile),
                )
            )
        except ValueError:
            continue

    out = pd.concat(frames, ignore_index=True)
    out_csv = (
        Path(args.out_csv).resolve()
        if args.out_csv
        else (root / "scripts" / "visualization" / "outputs" / "revote_decision_minimal_metrics.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Input: {csv_path}")
    print(f"Saved metrics: {out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
