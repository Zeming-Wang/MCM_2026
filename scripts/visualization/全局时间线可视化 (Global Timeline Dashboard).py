import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_risk_table(project_root: str) -> pd.DataFrame:
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")
    risk_path = os.path.join(out_dir, "elimination_risk_table.csv")
    if not os.path.exists(risk_path):
        raise FileNotFoundError(f"Missing {risk_path}. Run elimination risk script first.")
    df = pd.read_csv(risk_path, encoding="utf-8-sig")
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype(int)
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype(int)
    df["Elimination_Prob"] = pd.to_numeric(df["Elimination_Prob"], errors="coerce").fillna(0.0)
    df["CI05_FanVote"] = pd.to_numeric(df["CI05_FanVote"], errors="coerce").fillna(0.0)
    df["CI95_FanVote"] = pd.to_numeric(df["CI95_FanVote"], errors="coerce").fillna(0.0)
    df["V_Base"] = pd.to_numeric(df["V_Base"], errors="coerce").fillna(0.0)
    df["Delta_V"] = pd.to_numeric(df["Delta_V"], errors="coerce").fillna(0.0)
    df["Name"] = df["Name"].astype(str)
    return df


def _load_processed_wide(project_root: str) -> pd.DataFrame:
    processed_path = os.path.join(project_root, "data", "processed", "processed_mcm_wide_clean.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Missing {processed_path}. Generate processed data first.")
    df = pd.read_csv(processed_path, encoding="utf-8-sig")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    df["celebrity_name"] = df["celebrity_name"].astype(str)
    df["results"] = df["results"].astype(str)
    return df


def _build_actual_elim_map(df_wide: pd.DataFrame, max_weeks: int = 11) -> dict[tuple[int, int], str]:
    elim_map: dict[tuple[int, int], str] = {}
    seasons = sorted(df_wide["season"].dropna().astype(int).unique().tolist())
    for season in seasons:
        season_df = df_wide[df_wide["season"].astype(int) == int(season)].copy().reset_index(drop=True)
        for week in range(1, int(max_weeks) + 1):
            mask = season_df["results"].str.contains(
                f"Eliminated Week {int(week)}", case=False, na=False
            )
            if bool(mask.any()):
                name = str(season_df.loc[mask, "celebrity_name"].iloc[0])
                elim_map[(int(season), int(week))] = name
    return elim_map


def build_week_summary(df: pd.DataFrame, elim_map: dict[tuple[int, int], str]) -> pd.DataFrame:
    rows = []
    for (season, week), g in df.groupby(["Season", "Week"], sort=True):
        if g.empty:
            continue
        g = g.copy()
        g = g.sort_values("Elimination_Prob", ascending=False)
        top1 = float(g["Elimination_Prob"].head(1).mean())
        top3 = float(g["Elimination_Prob"].head(3).mean())
        ci_width = g["CI95_FanVote"].to_numpy(dtype=float) - g["CI05_FanVote"].to_numpy(dtype=float)
        ci_width = np.maximum(ci_width, 0.0)
        width_mean = float(np.mean(ci_width))
        width_p25 = float(np.quantile(ci_width, 0.25))
        width_p75 = float(np.quantile(ci_width, 0.75))
        base_scale = float(np.mean(np.abs(g["V_Base"].to_numpy(dtype=float))))
        resid_scale = float(np.mean(np.abs(g["Delta_V"].to_numpy(dtype=float))))
        influence_ratio = resid_scale / (base_scale + 1e-12)
        actual_elim_name = elim_map.get((int(season), int(week)))
        actual_elim_prob = np.nan
        if actual_elim_name is not None:
            hit = g[g["Name"] == str(actual_elim_name)]
            if not hit.empty:
                actual_elim_prob = float(hit["Elimination_Prob"].iloc[0])
        rows.append(
            {
                "Season": int(season),
                "Week": int(week),
                "Top1_Risk": top1,
                "Top3_Risk": top3,
                "CI_Width_Mean": width_mean,
                "CI_Width_P25": width_p25,
                "CI_Width_P75": width_p75,
                "Residual_Base_Ratio": influence_ratio,
                "N_Active": int(len(g)),
                "Actual_Elim_Prob": actual_elim_prob,
            }
        )
    df_sw = pd.DataFrame(rows)
    if df_sw.empty:
        return df_sw
    week_rows = []
    for week, g in df_sw.groupby("Week", sort=True):
        week_rows.append(
            {
                "Week": int(week),
                "Top1_Risk": float(g["Top1_Risk"].mean()),
                "Top3_Risk": float(g["Top3_Risk"].mean()),
                "CI_Width_Mean": float(g["CI_Width_Mean"].mean()),
                "CI_Width_P25": float(g["CI_Width_P25"].mean()),
                "CI_Width_P75": float(g["CI_Width_P75"].mean()),
                "Residual_Base_Ratio": float(g["Residual_Base_Ratio"].mean()),
                "N_Active": float(g["N_Active"].mean()),
                "Actual_Elim_Prob": float(np.nanmean(g["Actual_Elim_Prob"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(week_rows).sort_values("Week")


def plot_global_timeline(df_week: pd.DataFrame, out_path: str) -> str:
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12.6, 12.2), dpi=170, sharex=True)

    ax = axes[0]
    ax.plot(df_week["Week"], df_week["Top1_Risk"], marker="o", linewidth=2.4, color="#E76F51", label="Top-1 risk")
    ax.plot(df_week["Week"], df_week["Top3_Risk"], marker="s", linewidth=2.2, color="#2A9D8F", label="Top-3 avg risk")
    if "Actual_Elim_Prob" in df_week.columns:
        ax.plot(
            df_week["Week"],
            df_week["Actual_Elim_Prob"],
            linestyle="--",
            linewidth=2.1,
            color="#6C757D",
            label="Actual eliminated risk",
        )
    ax.set_ylabel("Elimination Risk")
    ax.set_title("Global Timeline: Elimination Risk")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)

    ax = axes[1]
    ax.plot(df_week["Week"], df_week["CI_Width_Mean"], marker="o", linewidth=2.4, color="#457B9D")
    ax.fill_between(
        df_week["Week"],
        df_week["CI_Width_P25"],
        df_week["CI_Width_P75"],
        color="#A8DADC",
        alpha=0.35,
    )
    ax.set_ylabel("Uncertainty Band Width")
    ax.set_title("Global Timeline: Fan Vote Uncertainty Width")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axes[2]
    ax.plot(
        df_week["Week"],
        df_week["Residual_Base_Ratio"],
        marker="o",
        linewidth=2.4,
        color="#8D99AE",
    )
    ax.set_ylabel("Residual/Base Ratio")
    ax.set_title("Global Timeline: Residual vs Base Influence")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axes[3]
    ax.plot(
        df_week["Week"],
        df_week["N_Active"],
        marker="o",
        linewidth=2.2,
        color="#264653",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Avg Active Contestants")
    ax.set_title("Global Timeline: Active Contestants")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    df_risk = _load_risk_table(project_root)
    df_wide = _load_processed_wide(project_root)
    elim_map = _build_actual_elim_map(df_wide, max_weeks=11)
    df_week = build_week_summary(df_risk, elim_map)
    if df_week.empty:
        raise RuntimeError("No weekly summary built from elimination risk table.")
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "global_timeline_summary.csv")
    df_week.to_csv(summary_path, index=False, encoding="utf-8-sig")
    out_path = os.path.join(out_dir, "global_timeline_dashboard.png")
    plot_global_timeline(df_week, out_path)
    print(f"Saved: {summary_path} (rows={len(df_week)})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
