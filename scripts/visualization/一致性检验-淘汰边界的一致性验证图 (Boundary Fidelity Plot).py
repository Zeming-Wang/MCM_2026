import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


def _rank_descending(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.argsort(np.argsort(-values)) + 1


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


def get_scoring_system(season: int) -> str:
    if int(season) <= 2 or int(season) >= 28:
        return "rank"
    return "percent"


def build_boundary_fidelity_table(df_wide: pd.DataFrame, df_pred: pd.DataFrame, max_weeks: int = 11) -> pd.DataFrame:
    df_pred = df_pred.copy()
    df_pred["Season"] = pd.to_numeric(df_pred["Season"], errors="coerce").astype(int)
    df_pred["Week"] = pd.to_numeric(df_pred["Week"], errors="coerce").astype(int)
    df_pred["Name"] = df_pred["Name"].astype(str)
    if "Fan_Vote_Percent" in df_pred.columns:
        df_pred["Fan_Vote_Percent"] = pd.to_numeric(df_pred["Fan_Vote_Percent"], errors="coerce").fillna(0.0)
    if "Fan_Vote_Rank" in df_pred.columns:
        df_pred["Fan_Vote_Rank"] = pd.to_numeric(df_pred["Fan_Vote_Rank"], errors="coerce").fillna(0.0)

    out_rows = []

    for season in sorted(pd.to_numeric(df_wide["season"], errors="coerce").dropna().astype(int).unique().tolist()):
        season_df = df_wide[df_wide["season"].astype(int) == int(season)].copy().reset_index(drop=True)
        if season_df.empty:
            continue

        for week in range(1, int(max_weeks) + 1):
            judge_cols = [c for c in _week_judge_cols(week) if c in season_df.columns]
            if not judge_cols:
                continue

            week_df = season_df[["celebrity_name", "results", "placement"] + judge_cols].copy()
            for col in judge_cols:
                week_df[col] = pd.to_numeric(week_df[col], errors="coerce").fillna(0.0)

            judge_points = week_df[judge_cols].sum(axis=1).to_numpy()
            active_mask = judge_points > 0
            week_df = week_df[active_mask].reset_index(drop=True)
            judge_points = judge_points[active_mask]
            if len(week_df) <= 1:
                continue

            names = week_df["celebrity_name"].astype(str).tolist()
            judge_rank = _rank_descending(judge_points)

            pred_g = df_pred[(df_pred["Season"] == int(season)) & (df_pred["Week"] == int(week))].copy()
            if pred_g.empty:
                continue
            pred_g = pred_g[pred_g["Name"].isin(names)].copy()
            if pred_g.empty:
                continue
            season_type = get_scoring_system(int(season))
            if season_type == "rank" and "Fan_Vote_Rank" in pred_g.columns:
                pred_map = dict(zip(pred_g["Name"].astype(str), pred_g["Fan_Vote_Rank"].astype(float)))
                pred_rank = np.asarray([pred_map.get(n, np.nan) for n in names], dtype=float)
                if np.isnan(pred_rank).any():
                    pred_rank = _rank_descending(
                        np.asarray([pred_g.set_index("Name")["Fan_Vote_Percent"].to_dict().get(n, 0.0) for n in names], dtype=float)
                    )
            else:
                if "Fan_Vote_Percent" not in pred_g.columns:
                    continue
                pred_map = dict(zip(pred_g["Name"].astype(str), pred_g["Fan_Vote_Percent"].astype(float)))
                pred_vals = np.asarray([pred_map.get(n, 0.0) for n in names], dtype=float)
                pred_vals = pred_vals / float(np.sum(pred_vals) if float(np.sum(pred_vals)) > 0 else 1.0)
                pred_rank = _rank_descending(pred_vals)

            elim_flag = (
                week_df["results"]
                .astype(str)
                .str.contains(f"Eliminated Week {int(week)}", case=False, na=False)
            )
            elim_name = None
            if bool(elim_flag.any()):
                elim_name = str(week_df.loc[elim_flag, "celebrity_name"].iloc[0])

            for i, name in enumerate(names):
                placement = pd.to_numeric(week_df.loc[i, "placement"], errors="coerce")
                if not np.isnan(placement) and int(placement) == 1:
                    status = "Winner"
                elif elim_name is not None and name == elim_name:
                    status = "Eliminated"
                else:
                    status = "Safe"

                out_rows.append(
                    {
                        "Season": int(season),
                        "Week": int(week),
                        "Name": name,
                        "SeasonType": season_type,
                        "JudgeRank": int(judge_rank[i]),
                        "PredictedRank": int(pred_rank[i]),
                        "Status": status,
                    }
                )

    return pd.DataFrame(out_rows)


def plot_boundary_fidelity_parallel_by_system(df_results: pd.DataFrame, output_path: str) -> str:
    if df_results.empty:
        raise ValueError("No boundary fidelity data to plot.")

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12.5, 9.0), dpi=170, sharex=True, sharey=True)

    panels = [("rank", "Rank System (Season ≤ 2 or ≥ 28)"), ("percent", "Percent System (Else)")]
    # Colors mimicking the network graph: Eliminated (Teal), Safe (Light Blue), Winner (Dark Blue)
    colors = ("#649b92", "#b3cde3", "#2b6a99")

    for ax, (stype, title) in zip(axes, panels):
        sub = df_results[df_results["SeasonType"] == stype].copy()
        if sub.empty:
            ax.set_axis_off()
            continue

        df_plot = sub[["JudgeRank", "PredictedRank", "Status"]].copy()
        # Reduce data to 1/4
        df_plot = df_plot.sample(frac=0.25, random_state=42)
        df_plot = df_plot.rename(columns={"Status": "EliminatedStatus"})

        parallel_coordinates(
            df_plot,
            "EliminatedStatus",
            color=colors,
            alpha=0.22,
            ax=ax,
        )
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Rank (1 is best)", fontsize=12, fontweight='bold')
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Boundary Fidelity: Judge Rank vs Predicted Rank (Split by Scoring System)", y=0.995, fontsize=16, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_path = os.path.join(project_root, "data", "processed", "processed_mcm_wide_clean.csv")
    raw_path = os.path.join(project_root, "data", "raw", "2026_MCM_Problem_C_Data.csv")
    pred_path = os.path.join(project_root, "data", "processed", "model1_fan_vote_predictions.csv")
    pred_path_fallback = os.path.join(project_root, "results", "model1_fan_vote_predictions_subset.csv")
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs", "Q1related")

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

    df_table = build_boundary_fidelity_table(df_wide, df_pred)
    df_table.to_csv(os.path.join(out_dir, "boundary_fidelity_table.csv"), index=False, encoding="utf-8-sig")
    out_path = os.path.join(out_dir, "boundary_fidelity_parallel_by_system.png")
    print(f"Generating plot to: {out_path}")
    plot_boundary_fidelity_parallel_by_system(df_table, out_path)
    if not os.path.exists(out_path):
        print(f"Error: File was not created at {out_path}")
        return
    print(f"Saved: {out_path} (rows={len(df_table)})")

    # Copy to charts directory
    charts_dir = os.path.join(project_root, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    chart_dest = os.path.join(charts_dir, "boundary_fidelity_parallel_by_system.png")
    shutil.copy(out_path, chart_dest)
    print(f"Copied to: {chart_dest}")


if __name__ == "__main__":
    main()
