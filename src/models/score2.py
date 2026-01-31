import os
import numpy as np
import pandas as pd


def _week_judge_cols(week: int) -> list[str]:
    return [
        f"week{week}_judge1_score",
        f"week{week}_judge2_score",
        f"week{week}_judge3_score",
        f"week{week}_judge4_score",
    ]


def _rank_descending(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.argsort(np.argsort(-values)) + 1


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build_rank_sum_results(
    processed_path: str,
    fan_pred_path: str,
    out_path: str,
    out_summary_path: str,
    max_weeks: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Missing processed data: {processed_path}")
    if not os.path.exists(fan_pred_path):
        raise FileNotFoundError(f"Missing fan prediction data: {fan_pred_path}")

    df = pd.read_csv(processed_path, encoding="utf-8-sig")
    need_cols = {"season", "celebrity_name", "results"}
    miss = sorted(need_cols - set(df.columns))
    if miss:
        raise ValueError(f"Missing required columns in processed file: {miss}")

    fan = pd.read_csv(fan_pred_path, encoding="utf-8-sig")
    fan_need = {"Season", "Week", "Name", "Fan_Vote_Percent", "Fan_Vote_Rank"}
    fan_miss = sorted(fan_need - set(fan.columns))
    if fan_miss:
        raise ValueError(f"Missing required columns in fan prediction file: {fan_miss}")

    fan = fan.copy()
    fan["Season"] = pd.to_numeric(fan["Season"], errors="coerce").astype("Int64")
    fan["Week"] = pd.to_numeric(fan["Week"], errors="coerce").astype("Int64")
    fan["Name"] = fan["Name"].astype(str)

    seasons = (
        pd.to_numeric(df["season"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    seasons = sorted(seasons)

    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for season in seasons:
        season_df = df[df["season"] == season].copy().reset_index(drop=True)

        for week in range(1, max_weeks + 1):
            judge_cols = [c for c in _week_judge_cols(week) if c in season_df.columns]
            if not judge_cols:
                continue

            week_df = season_df[["celebrity_name", "results"] + judge_cols].copy()
            for col in judge_cols:
                week_df[col] = pd.to_numeric(week_df[col], errors="coerce").fillna(0.0)

            judge_points = week_df[judge_cols].sum(axis=1).to_numpy(dtype=float)
            active_mask = judge_points > 0
            week_df = week_df[active_mask].reset_index(drop=True)
            judge_points = judge_points[active_mask]

            if len(week_df) < 2:
                continue

            judge_rank = _rank_descending(judge_points)
            s = float(np.sum(judge_points))
            judge_percent = judge_points / (s if s > 0 else 1.0)

            elim_flag = (
                week_df["results"]
                .astype(str)
                .str.contains(f"Eliminated Week {week}", case=False, na=False)
                .astype(int)
                .to_numpy()
            )
            actual_elim_names = (
                week_df.loc[elim_flag.astype(bool), "celebrity_name"].astype(str).tolist()
            )
            actual_elim_name = actual_elim_names[0] if len(actual_elim_names) > 0 else ""

            week_fan = fan[(fan["Season"] == season) & (fan["Week"] == week)].copy()
            if week_fan.empty:
                continue

            merged = pd.merge(
                week_df.assign(
                    Season=int(season),
                    Week=int(week),
                    Name=week_df["celebrity_name"].astype(str),
                    Judge_Points=judge_points,
                    Judge_Rank=judge_rank,
                    Judge_Percent=judge_percent,
                ),
                week_fan[["Season", "Week", "Name", "Fan_Vote_Percent", "Fan_Vote_Rank"]],
                on=["Season", "Week", "Name"],
                how="inner",
            )

            if len(merged) < 2:
                continue

            merged["Fan_Vote_Percent"] = pd.to_numeric(
                merged["Fan_Vote_Percent"], errors="coerce"
            ).fillna(0.0)
            merged["Fan_Vote_Rank"] = pd.to_numeric(
                merged["Fan_Vote_Rank"], errors="coerce"
            ).fillna(0.0)
            merged["Judge_Rank"] = pd.to_numeric(merged["Judge_Rank"], errors="coerce").fillna(
                0.0
            )

            merged["Combined_Rank_Sum"] = merged["Judge_Rank"] + merged["Fan_Vote_Rank"]
            merged["Combined_Rank"] = (
                merged["Combined_Rank_Sum"].rank(method="dense", ascending=True).astype(int)
            )

            worst_idx = int(np.argmax(merged["Combined_Rank_Sum"].to_numpy(dtype=float)))
            predicted_elim_name = str(merged.iloc[worst_idx]["Name"])

            merged["Actual_Eliminated_Name"] = actual_elim_name
            merged["Predicted_Eliminated_Name"] = predicted_elim_name
            merged["Predicted_Eliminated_Flag"] = (merged["Name"] == predicted_elim_name).astype(
                int
            )
            merged["Actual_Eliminated_Flag"] = (merged["Name"] == actual_elim_name).astype(int)

            cols_out = [
                "Season",
                "Week",
                "Name",
                "Judge_Points",
                "Judge_Rank",
                "Judge_Percent",
                "Fan_Vote_Percent",
                "Fan_Vote_Rank",
                "Combined_Rank_Sum",
                "Combined_Rank",
                "Actual_Eliminated_Name",
                "Predicted_Eliminated_Name",
                "Actual_Eliminated_Flag",
                "Predicted_Eliminated_Flag",
            ]
            all_rows.append(merged[cols_out].copy())

            corr_fan = _safe_corr(
                merged["Combined_Rank"].to_numpy(dtype=float),
                merged["Fan_Vote_Rank"].to_numpy(dtype=float),
            )
            corr_judge = _safe_corr(
                merged["Combined_Rank"].to_numpy(dtype=float),
                merged["Judge_Rank"].to_numpy(dtype=float),
            )
            summary_rows.append(
                {
                    "Season": int(season),
                    "Week": int(week),
                    "N": int(len(merged)),
                    "Actual_Eliminated_Name": actual_elim_name,
                    "Predicted_Eliminated_Name": predicted_elim_name,
                    "Hit_Elimination": int(
                        actual_elim_name != "" and predicted_elim_name == actual_elim_name
                    ),
                    "Corr_Combined_vs_FanRank": corr_fan,
                    "Corr_Combined_vs_JudgeRank": corr_judge,
                    "Leaning_FanRank": int(
                        (not np.isnan(corr_fan))
                        and (not np.isnan(corr_judge))
                        and (corr_fan > corr_judge)
                    ),
                }
            )

    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
    else:
        out_df = pd.DataFrame(
            columns=[
                "Season",
                "Week",
                "Name",
                "Judge_Points",
                "Judge_Rank",
                "Judge_Percent",
                "Fan_Vote_Percent",
                "Fan_Vote_Rank",
                "Combined_Rank_Sum",
                "Combined_Rank",
                "Actual_Eliminated_Name",
                "Predicted_Eliminated_Name",
                "Actual_Eliminated_Flag",
                "Predicted_Eliminated_Flag",
            ]
        )

    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)
    summary_df.to_csv(out_summary_path, index=False, encoding="utf-8-sig")

    return out_df, summary_df


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(
        os.environ.get("MCM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    ).resolve()

    PROCESSED_PATH = str(project_root / "data" / "processed" / "processed_mcm_wide_clean.csv")
    FAN_PRED_PATH = str(project_root / "data" / "processed" / "model1_fan_vote_predictions.csv")

    OUT_PATH = str(project_root / "data" / "processed" / "model1_rank_sum_scoring.csv")
    OUT_SUMMARY_PATH = str(project_root / "data" / "processed" / "model1_rank_sum_week_summary.csv")

    out_df, summary_df = build_rank_sum_results(
        processed_path=PROCESSED_PATH,
        fan_pred_path=FAN_PRED_PATH,
        out_path=OUT_PATH,
        out_summary_path=OUT_SUMMARY_PATH,
        max_weeks=11,
    )

    hit = int(summary_df["Hit_Elimination"].sum()) if not summary_df.empty else 0
    total = int(len(summary_df))
    print(f"Saved: {OUT_PATH} (rows={len(out_df)})")
    print(f"Saved: {OUT_SUMMARY_PATH} (rows={len(summary_df)})")
    print(f"Elimination hit rate (week-level): {hit}/{total}" if total > 0 else "No weeks scored.")