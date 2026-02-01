import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_scoring_system(season: int) -> str:
    if season <= 2 or season >= 28:
        return "rank"
    return "percent"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - float(np.max(x))
    ex = np.exp(x)
    s = float(np.sum(ex))
    return ex / (s if s > 0 else 1.0)


def _rank_descending(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.argsort(np.argsort(-values)) + 1


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        v = v.reshape(-1)
    n = int(v.shape[0])
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        theta = 0.0
    else:
        rho = int(rho[-1])
        theta = float((cssv[rho] - 1.0) / (rho + 1))
    w = np.maximum(v - theta, 0.0)
    s = float(np.sum(w))
    return w / (s if s > 0 else 1.0)


def _project_to_halfspace(p: np.ndarray, a: np.ndarray, c: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    ap = float(np.dot(a, p))
    if ap <= c:
        return p
    denom = float(np.dot(a, a))
    if denom <= 0:
        return p
    return p - ((ap - c) / denom) * a


def project_percent_elimination_feasible(
    p_raw: np.ndarray,
    p_judge: np.ndarray,
    eliminated_idx: int,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> np.ndarray:
    p_raw = _project_to_simplex(p_raw)
    p_judge = np.asarray(p_judge, dtype=float)
    n = int(p_raw.shape[0])
    e = int(eliminated_idx)
    if n <= 1 or e < 0 or e >= n:
        return p_raw

    a_list = []
    c_list = []
    for i in range(n):
        if i == e:
            continue
        a = np.zeros(n, dtype=float)
        a[e] = 1.0
        a[i] = -1.0
        c = float(p_judge[i] - p_judge[e])
        a_list.append(a)
        c_list.append(c)

    p = p_raw.copy()
    for _ in range(int(max_iter)):
        p_prev = p.copy()
        for a, c in zip(a_list, c_list):
            p = _project_to_halfspace(p, a, c)
        p = _project_to_simplex(p)
        if float(np.linalg.norm(p - p_prev)) < float(tol):
            break
    return p


def project_rank_elimination_feasible(
    p_raw: np.ndarray,
    judge_ranks: np.ndarray,
    eliminated_idx: int,
    seed: int = 0,
) -> np.ndarray:
    p_raw = _project_to_simplex(p_raw)
    judge_ranks = np.asarray(judge_ranks, dtype=int)
    n = int(p_raw.shape[0])
    e = int(eliminated_idx)
    if n <= 1 or e < 0 or e >= n:
        return p_raw

    rng = np.random.default_rng(int(seed))

    def feasible(p: np.ndarray) -> bool:
        r_fan = _rank_descending(p).astype(int)
        r_sum = judge_ranks + r_fan
        return bool(np.all(r_sum[e] >= np.delete(r_sum, e)))

    best = None
    best_dist = float("inf")

    for concentration, draws in [(10.0, 2000), (30.0, 4000), (100.0, 6000)]:
        alpha = 1.0 + float(concentration) * p_raw
        for _ in range(int(draws)):
            cand = rng.dirichlet(alpha)
            if not feasible(cand):
                continue
            dist = float(np.linalg.norm(cand - p_raw))
            if dist < best_dist:
                best_dist = dist
                best = cand
                if best_dist < 1e-6:
                    return best

    return best if best is not None else p_raw


def _parse_vector_cell(x, expected_len: int | None = None) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        v = np.asarray(x, dtype=float).reshape(-1)
        return v
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.asarray([], dtype=float)
    s = str(x).strip()
    if s == "":
        return np.asarray([], dtype=float)
    s = s.strip("()")
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if s == "":
        return np.asarray([], dtype=float)
    parts = re.split(r"[\s,]+", s)
    vals = []
    for p in parts:
        if p == "" or p.lower() in {"nan", "none"}:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            continue
    v = np.asarray(vals, dtype=float).reshape(-1)
    if expected_len is not None and expected_len > 0 and v.size != expected_len:
        if v.size > expected_len:
            v = v[:expected_len]
    return v


def _ensure_processed_wide(processed_path: str, raw_path: str) -> pd.DataFrame:
    if os.path.exists(processed_path):
        return pd.read_csv(processed_path, encoding="utf-8-sig")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from scripts.preprocessing.process_data import clean_raw_to_wide_clean

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    df_wide = clean_raw_to_wide_clean(df_raw)
    df_wide.to_csv(processed_path, index=False, encoding="utf-8-sig")
    return df_wide


def _week_judge_cols(week: int) -> list[str]:
    return [
        f"week{week}_judge1_score",
        f"week{week}_judge2_score",
        f"week{week}_judge3_score",
        f"week{week}_judge4_score",
    ]


@dataclass(frozen=True)
class WeekContext:
    season: int
    week: int
    names: list[str]
    judge_points: np.ndarray
    judge_percent: np.ndarray
    judge_ranks: np.ndarray
    eliminated_name: str | None


def build_week_context(df_wide: pd.DataFrame, season: int, week: int) -> WeekContext | None:
    season_df = df_wide[df_wide["season"].astype(int) == int(season)].copy().reset_index(drop=True)
    if season_df.empty:
        return None
    judge_cols = [c for c in _week_judge_cols(int(week)) if c in season_df.columns]
    if not judge_cols:
        return None

    week_df = season_df[["celebrity_name", "results"] + judge_cols].copy()
    for col in judge_cols:
        week_df[col] = pd.to_numeric(week_df[col], errors="coerce").fillna(0.0)
    judge_points = week_df[judge_cols].sum(axis=1).to_numpy()
    active_mask = judge_points > 0
    week_df = week_df[active_mask].reset_index(drop=True)
    judge_points = judge_points[active_mask]
    if len(week_df) <= 1:
        return None

    names = week_df["celebrity_name"].astype(str).tolist()
    s = float(np.sum(judge_points))
    judge_percent = judge_points / (s if s > 0 else 1.0)
    judge_ranks = _rank_descending(judge_points).astype(int)

    elim_flag = (
        week_df["results"]
        .astype(str)
        .str.contains(f"Eliminated Week {int(week)}", case=False, na=False)
    )
    eliminated_name = None
    if bool(elim_flag.any()):
        eliminated_name = str(week_df.loc[elim_flag, "celebrity_name"].iloc[0])

    return WeekContext(
        season=int(season),
        week=int(week),
        names=names,
        judge_points=np.asarray(judge_points, dtype=float),
        judge_percent=np.asarray(judge_percent, dtype=float),
        judge_ranks=np.asarray(judge_ranks, dtype=int),
        eliminated_name=eliminated_name,
    )


def compute_projection_artifacts(
    df_wide: pd.DataFrame,
    df_pred: pd.DataFrame,
    alpha: float = 0.5,
) -> tuple[pd.DataFrame, list[dict]]:
    required = {"Season", "Week", "Name", "Fan_Vote_Percent", "Delta_V"}
    missing = sorted(required - set(df_pred.columns))
    if missing:
        raise ValueError(f"Missing required columns in predictions: {missing}")

    df_pred = df_pred.copy()
    df_pred["Season"] = pd.to_numeric(df_pred["Season"], errors="coerce").astype(int)
    df_pred["Week"] = pd.to_numeric(df_pred["Week"], errors="coerce").astype(int)
    df_pred["Name"] = df_pred["Name"].astype(str)
    df_pred["Fan_Vote_Percent"] = pd.to_numeric(df_pred["Fan_Vote_Percent"], errors="coerce")

    season_week_records = []
    summary_rows = []

    for (season, week), g in df_pred.groupby(["Season", "Week"], sort=True):
        ctx = build_week_context(df_wide, int(season), int(week))
        if ctx is None or ctx.eliminated_name is None:
            continue

        g = g.sort_values("Name").reset_index(drop=True)
        pred_names = g["Name"].astype(str).tolist()
        if ctx.eliminated_name not in pred_names:
            continue
        e = int(pred_names.index(ctx.eliminated_name))

        ctx_idx = {str(n): i for i, n in enumerate(ctx.names)}
        if any(n not in ctx_idx for n in pred_names):
            continue
        reorder = np.asarray([ctx_idx[n] for n in pred_names], dtype=int)
        judge_points = np.asarray(ctx.judge_points, dtype=float)[reorder]
        judge_percent = np.asarray(ctx.judge_percent, dtype=float)[reorder]
        judge_ranks = np.asarray(ctx.judge_ranks, dtype=int)[reorder]

        delta_v = pd.to_numeric(g["Delta_V"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if delta_v.size != len(g):
            delta_v = np.zeros(len(g), dtype=float)

        p_raw = _softmax(float(alpha) * delta_v)

        season_type = get_scoring_system(int(season))
        if season_type == "percent":
            p_proj = project_percent_elimination_feasible(p_raw, judge_percent, e)
            combined_raw = judge_percent + p_raw
            combined_proj = judge_percent + p_proj
            margins_raw = combined_raw[e] - np.delete(combined_raw, e)
            margins_proj = combined_proj[e] - np.delete(combined_proj, e)
        else:
            p_proj = project_rank_elimination_feasible(p_raw, judge_ranks, e)
            r_fan_raw = _rank_descending(p_raw).astype(int)
            r_fan_proj = _rank_descending(p_proj).astype(int)
            r_sum_raw = judge_ranks + r_fan_raw
            r_sum_proj = judge_ranks + r_fan_proj
            margins_raw = np.delete(r_sum_raw, e) - r_sum_raw[e]
            margins_proj = np.delete(r_sum_proj, e) - r_sum_proj[e]

        p_fused = g["Fan_Vote_Percent"].to_numpy(dtype=float)
        p_fused = _project_to_simplex(p_fused)
        if season_type == "percent":
            combined_fused = judge_percent + p_fused
            margins_fused = combined_fused[e] - np.delete(combined_fused, e)
        else:
            r_fan_fused = _rank_descending(p_fused).astype(int)
            r_sum_fused = judge_ranks + r_fan_fused
            margins_fused = np.delete(r_sum_fused, e) - r_sum_fused[e]

        offset = np.abs(p_proj - p_raw)
        offset_l1 = float(np.sum(offset))
        offset_l2 = float(np.linalg.norm(p_proj - p_raw))

        summary_rows.append(
            {
                "Season": int(season),
                "Week": int(week),
                "SeasonType": season_type,
                "EliminatedName": ctx.eliminated_name,
                "ProjectionOffset_L1": offset_l1,
                "ProjectionOffset_L2": offset_l2,
                "MaxMarginRaw": float(np.max(margins_raw)) if len(margins_raw) else np.nan,
                "MaxMarginProj": float(np.max(margins_proj)) if len(margins_proj) else np.nan,
                "MaxMarginFused": float(np.max(margins_fused)) if len(margins_fused) else np.nan,
            }
        )

        season_week_records.append(
            {
                "season": int(season),
                "week": int(week),
                "names": pred_names,
                "eliminated_idx": e,
                "eliminated_name": ctx.eliminated_name,
                "season_type": season_type,
                "judge_points": judge_points,
                "judge_percent": judge_percent,
                "judge_ranks": judge_ranks,
                "delta_v": delta_v,
                "p_raw": p_raw,
                "p_proj": p_proj,
                "p_fused": p_fused,
                "offset": offset,
                "margins_raw": margins_raw,
                "margins_proj": margins_proj,
                "margins_fused": margins_fused,
            }
        )

    return pd.DataFrame(summary_rows), season_week_records


def plot_season_week_projection(record: dict, out_dir: str) -> str:
    season = int(record["season"])
    week = int(record["week"])
    names = list(record["names"])
    e = int(record["eliminated_idx"])
    season_type = str(record["season_type"])

    others = [n for i, n in enumerate(names) if i != e]
    margins = pd.DataFrame(
        {
            "raw": np.asarray(record["margins_raw"], dtype=float),
            "projected": np.asarray(record["margins_proj"], dtype=float),
            "fused": np.asarray(record["margins_fused"], dtype=float),
        },
        index=others,
    )
    margins = margins.loc[margins.mean(axis=1).sort_values(ascending=False).index]
    offsets = pd.Series(np.asarray(record["offset"], dtype=float), index=names)

    plt.style.use("seaborn-v0_8-paper")
    fig = plt.figure(figsize=(13.5, 7.5), dpi=160)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 0.65])

    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        margins.T,
        cmap="RdBu_r",
        center=0.0,
        cbar_kws={"label": "Elimination Margin (positive = violation / inconsistency)"},
        ax=ax0,
    )
    ax0.set_title(f"Season {season} Week {week} Margin Heatmap ({season_type})")
    ax0.set_xlabel("Contestant (excluding eliminated)")
    ax0.set_ylabel("State")

    ax1 = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        offsets.to_frame("Projection |Δp|").T,
        cmap="viridis",
        cbar_kws={"label": "Probability Correction Magnitude"},
        ax=ax1,
    )
    ax1.set_xlabel("Contestant")
    ax1.set_ylabel("")

    ax2 = fig.add_subplot(gs[:, 1])
    p_raw = np.asarray(record["p_raw"], dtype=float)
    p_proj = np.asarray(record["p_proj"], dtype=float)
    colors = np.full(len(names), 0.5)
    colors[e] = 0.0
    sc = ax2.scatter(p_raw, p_proj, c=offsets.to_numpy(), cmap="viridis", s=90, edgecolor="white", linewidth=0.6)
    lim = float(max(np.max(p_raw), np.max(p_proj), 1e-6))
    ax2.plot([0, lim], [0, lim], linestyle="--", color="#CC3D3D", linewidth=1.4, alpha=0.9)
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.set_xlabel("Raw Fan Probability (softmax(α·ΔV))")
    ax2.set_ylabel("Projected Fan Probability (P(raw))")
    ax2.set_title(f"Projection Fidelity (Eliminated: {record['eliminated_name']})")
    cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("|Δp|")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"projection_consistency_S{season}_W{week}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_projection_offset_summary(df_summary: pd.DataFrame, out_dir: str) -> str:
    if df_summary.empty:
        raise ValueError("No summary rows to plot.")
    pivot = df_summary.pivot_table(
        index="Season",
        columns="Week",
        values="ProjectionOffset_L1",
        aggfunc="mean",
    ).sort_index()

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=160)
    sns.heatmap(
        pivot,
        cmap="mako",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "ProjectionOffset_L1 (higher = stronger correction)"},
        ax=ax,
    )
    ax.set_title("Projection Offset Summary Heatmap")
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "projection_offset_summary_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    processed_path = os.path.join(project_root, "data", "processed", "processed_mcm_wide_clean.csv")
    raw_path = os.path.join(project_root, "data", "raw", "2026_MCM_Problem_C_Data.csv")
    pred_path = os.path.join(project_root, "data", "processed", "model1_fan_vote_predictions.csv")
    pred_path_fallback = os.path.join(project_root, "results", "model1_fan_vote_predictions_subset.csv")
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")

    df_wide = _ensure_processed_wide(processed_path=processed_path, raw_path=raw_path)

    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    elif os.path.exists(pred_path_fallback):
        df_pred = pd.read_csv(pred_path_fallback, encoding="utf-8-sig")
    else:
        raise FileNotFoundError(
            "Missing predictions CSV. Run src/models/model1_prediction_newest_canrun.py to generate "
            f"{pred_path} (or run subset script to generate {pred_path_fallback})."
        )

    df_summary, records = compute_projection_artifacts(df_wide, df_pred, alpha=0.5)
    if df_summary.empty or not records:
        raise RuntimeError("No season-week records found for plotting (check predictions and data alignment).")

    summary_path = plot_projection_offset_summary(df_summary, out_dir=out_dir)

    saved = 0
    for rec in records[:40]:
        plot_season_week_projection(rec, out_dir=out_dir)
        saved += 1

    df_summary.to_csv(os.path.join(out_dir, "projection_consistency_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved summary heatmap: {summary_path}")
    print(f"Saved {saved} season-week figures to: {out_dir}")


if __name__ == "__main__":
    main()
