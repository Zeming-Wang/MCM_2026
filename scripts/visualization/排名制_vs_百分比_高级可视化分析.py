import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_summary(path: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")
    df["N"] = pd.to_numeric(df["N"], errors="coerce").fillna(0).astype(int)
    df["Hit_Elimination"] = pd.to_numeric(df["Hit_Elimination"], errors="coerce").fillna(0).astype(int)
    df["Actual_Eliminated_Name"] = df["Actual_Eliminated_Name"].astype(str)
    df["Predicted_Eliminated_Name"] = df["Predicted_Eliminated_Name"].astype(str)
    if method == "rank":
        df["Corr_Fan"] = pd.to_numeric(df["Corr_Combined_vs_FanRank"], errors="coerce")
        df["Corr_Judge"] = pd.to_numeric(df["Corr_Combined_vs_JudgeRank"], errors="coerce")
        df["Leaning_Fan"] = pd.to_numeric(df["Leaning_FanRank"], errors="coerce").fillna(0).astype(int)
    else:
        df["Corr_Fan"] = pd.to_numeric(df["Corr_Combined_vs_FanPercent"], errors="coerce")
        df["Corr_Judge"] = pd.to_numeric(df["Corr_Combined_vs_JudgePercent"], errors="coerce")
        df["Leaning_Fan"] = pd.to_numeric(df["Leaning_FanPercent"], errors="coerce").fillna(0).astype(int)
    df["Method"] = method
    return df[
        [
            "Season",
            "Week",
            "N",
            "Actual_Eliminated_Name",
            "Predicted_Eliminated_Name",
            "Hit_Elimination",
            "Corr_Fan",
            "Corr_Judge",
            "Leaning_Fan",
            "Method",
        ]
    ].copy()


def _load_scoring(path: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")
    df["Name"] = df["Name"].astype(str)
    df["Predicted_Eliminated_Name"] = df["Predicted_Eliminated_Name"].astype(str)
    df["Method"] = method
    return df


def build_comparison_tables(rank_sum: pd.DataFrame, percent_sum: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([rank_sum, percent_sum], ignore_index=True)
    combined["Fan_Leaning_Score"] = combined["Corr_Fan"] - combined["Corr_Judge"]
    return combined


def build_flow_table(rank_score: pd.DataFrame, percent_score: pd.DataFrame) -> pd.DataFrame:
    rank_pred = (
        rank_score.drop_duplicates(subset=["Season", "Week"])[
            ["Season", "Week", "Predicted_Eliminated_Name"]
        ]
        .rename(columns={"Predicted_Eliminated_Name": "Rank_Predicted"})
        .copy()
    )
    percent_pred = (
        percent_score.drop_duplicates(subset=["Season", "Week"])[
            ["Season", "Week", "Predicted_Eliminated_Name"]
        ]
        .rename(columns={"Predicted_Eliminated_Name": "Percent_Predicted"})
        .copy()
    )
    merged = pd.merge(rank_pred, percent_pred, on=["Season", "Week"], how="inner")
    flow = (
        merged.groupby(["Rank_Predicted", "Percent_Predicted"], dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    return flow


def plot_raincloud(df: pd.DataFrame, out_path: str) -> str:
    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10.6, 6.4), dpi=180)
    order = ["rank", "percent"]
    palette = {"rank": "#3A86FF", "percent": "#FF006E"}
    sns.violinplot(
        data=df,
        x="Method",
        y="Fan_Leaning_Score",
        order=order,
        hue="Method",
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0,
        ax=ax,
        alpha=0.6,
        legend=False,
        dodge=False,
    )
    sns.boxplot(
        data=df,
        x="Method",
        y="Fan_Leaning_Score",
        order=order,
        width=0.22,
        ax=ax,
        color="#1F1F1F",
        fliersize=0,
        boxprops={"alpha": 0.5},
        whiskerprops={"alpha": 0.7},
        medianprops={"color": "white", "linewidth": 2},
    )
    sns.stripplot(
        data=df,
        x="Method",
        y="Fan_Leaning_Score",
        order=order,
        color="white",
        edgecolor="#3C3C3C",
        linewidth=0.4,
        size=4.2,
        alpha=0.8,
        ax=ax,
    )
    ax.axhline(0.0, color="#5C5C5C", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Method")
    ax.set_ylabel("Fan Leaning Score (Corr_Fan - Corr_Judge)")
    ax.set_title("Raincloud: Fan Leaning Score Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_quadrant(df: pd.DataFrame, out_path: str) -> str:
    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(9.8, 7.4), dpi=180)
    palette = {"rank": "#3A86FF", "percent": "#FF006E"}
    for method, g in df.groupby("Method", sort=True):
        ax.scatter(
            g["Corr_Fan"],
            g["Corr_Judge"],
            s=60,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            label=method,
            color=palette.get(method, "#444444"),
        )
    lim = float(
        max(
            1.0,
            np.nanmax(np.abs(df["Corr_Fan"].to_numpy(dtype=float))),
            np.nanmax(np.abs(df["Corr_Judge"].to_numpy(dtype=float))),
        )
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0.0, color="#A0A0A0", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color="#A0A0A0", linewidth=1.0, alpha=0.7)
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="#222222", linewidth=1.1, alpha=0.8)
    ax.set_xlabel("Corr(Combined, Fan)")
    ax.set_ylabel("Corr(Combined, Judge)")
    ax.set_title("Quadrant: Fan vs Judge Alignment")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _polar_to_cart(r: float, theta: float) -> tuple[float, float]:
    return float(r * np.cos(theta)), float(r * np.sin(theta))


def plot_chord(flow: pd.DataFrame, out_path: str, top_n: int = 12) -> str:
    flow = flow.copy()
    flow["Count"] = pd.to_numeric(flow["Count"], errors="coerce").fillna(0.0)
    total_by_name = (
        pd.concat(
            [
                flow[["Rank_Predicted", "Count"]].rename(columns={"Rank_Predicted": "Name"}),
                flow[["Percent_Predicted", "Count"]].rename(columns={"Percent_Predicted": "Name"}),
            ],
            ignore_index=True,
        )
        .groupby("Name", dropna=False)["Count"]
        .sum()
        .sort_values(ascending=False)
    )
    top_names = total_by_name.head(int(top_n)).index.tolist()
    flow["Rank_Predicted"] = flow["Rank_Predicted"].where(flow["Rank_Predicted"].isin(top_names), "Other")
    flow["Percent_Predicted"] = flow["Percent_Predicted"].where(
        flow["Percent_Predicted"].isin(top_names), "Other"
    )
    flow = (
        flow.groupby(["Rank_Predicted", "Percent_Predicted"], dropna=False)["Count"]
        .sum()
        .reset_index()
    )
    names = sorted(set(flow["Rank_Predicted"].tolist() + flow["Percent_Predicted"].tolist()))
    n = len(names)
    if n == 0:
        raise RuntimeError("No flow records available for chord diagram.")
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + np.pi / 2.0
    radius = 1.0
    inner = 0.86
    palette = sns.color_palette("tab20", n_colors=max(n, 3))
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(names)}

    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(11.2, 11.2), dpi=220)
    ax.set_aspect("equal")
    ax.axis("off")
    for i, name in enumerate(names):
        theta = angles[i]
        x, y = _polar_to_cart(radius, theta)
        rot = np.degrees(theta) - 90.0
        if 90.0 < (rot % 360.0) < 270.0:
            rot += 180.0
            ha = "right"
        else:
            ha = "left"
        ax.text(
            x * 1.05,
            y * 1.05,
            name,
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=9.6,
            color="#222222",
        )
        ax.plot([x * 0.98, x * 1.02], [y * 0.98, y * 1.02], color=color_map[name], lw=2.0)

    max_count = float(flow["Count"].max()) if not flow.empty else 1.0
    for _, row in flow.iterrows():
        a = row["Rank_Predicted"]
        b = row["Percent_Predicted"]
        if a == b:
            continue
        if float(row["Count"]) <= 0:
            continue
        ia = names.index(a)
        ib = names.index(b)
        ta = angles[ia]
        tb = angles[ib]
        start = np.array(_polar_to_cart(inner, ta))
        end = np.array(_polar_to_cart(inner, tb))
        c1 = start * 0.2
        c2 = end * 0.2
        verts = [start, c1, c2, end]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        lw = 0.8 + 4.0 * float(row["Count"]) / max_count
        alpha = 0.15 + 0.7 * float(row["Count"]) / max_count
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=color_map.get(a, "#555555"),
            linewidth=lw,
            alpha=alpha,
        )
        ax.add_patch(patch)

    ax.add_patch(Circle((0.0, 0.0), radius=inner, fill=False, edgecolor="#D0D0D0", linewidth=1.0))
    ax.set_title("Chord Diagram: Rank vs Percent Elimination Predictions")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("Method", sort=True):
        hit_rate = float(g["Hit_Elimination"].mean()) if len(g) else 0.0
        rows.append(
            {
                "Method": method,
                "Weeks": int(len(g)),
                "Hit_Rate": hit_rate,
                "Corr_Fan_Mean": float(np.nanmean(g["Corr_Fan"].to_numpy(dtype=float))),
                "Corr_Judge_Mean": float(np.nanmean(g["Corr_Judge"].to_numpy(dtype=float))),
                "Leaning_Fan_Ratio": float(g["Leaning_Fan"].mean()) if len(g) else 0.0,
                "Fan_Leaning_Score_Mean": float(
                    np.nanmean(g["Fan_Leaning_Score"].to_numpy(dtype=float))
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data", "processed")
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    rank_summary_path = os.path.join(data_dir, "model1_rank_sum_week_summary.csv")
    percent_summary_path = os.path.join(data_dir, "model1_percent_sum_week_summary.csv")
    rank_score_path = os.path.join(data_dir, "model1_rank_sum_scoring.csv")
    percent_score_path = os.path.join(data_dir, "model1_percent_sum_scoring.csv")

    if not os.path.exists(rank_summary_path):
        raise FileNotFoundError(f"Missing {rank_summary_path}")
    if not os.path.exists(percent_summary_path):
        raise FileNotFoundError(f"Missing {percent_summary_path}")
    if not os.path.exists(rank_score_path):
        raise FileNotFoundError(f"Missing {rank_score_path}")
    if not os.path.exists(percent_score_path):
        raise FileNotFoundError(f"Missing {percent_score_path}")

    rank_summary = _load_summary(rank_summary_path, "rank")
    percent_summary = _load_summary(percent_summary_path, "percent")
    rank_score = _load_scoring(rank_score_path, "rank")
    percent_score = _load_scoring(percent_score_path, "percent")

    combined = build_comparison_tables(rank_summary, percent_summary)
    combined.to_csv(os.path.join(out_dir, "rank_percent_week_summary.csv"), index=False, encoding="utf-8-sig")

    summary_stats = build_summary_stats(combined)
    summary_stats.to_csv(os.path.join(out_dir, "rank_percent_summary_stats.csv"), index=False, encoding="utf-8-sig")

    flow = build_flow_table(rank_score, percent_score)
    flow.to_csv(os.path.join(out_dir, "rank_percent_prediction_flow.csv"), index=False, encoding="utf-8-sig")

    raincloud_path = os.path.join(out_dir, "raincloud_fan_leaning_score.png")
    quadrant_path = os.path.join(out_dir, "quadrant_corr_alignment.png")
    chord_path = os.path.join(out_dir, "chord_rank_vs_percent_predictions.png")

    plot_raincloud(combined, raincloud_path)
    plot_quadrant(combined, quadrant_path)
    plot_chord(flow, chord_path)

    print(f"Saved: {raincloud_path}")
    print(f"Saved: {quadrant_path}")
    print(f"Saved: {chord_path}")
    print(f"Saved: {os.path.join(out_dir, 'rank_percent_week_summary.csv')}")
    print(f"Saved: {os.path.join(out_dir, 'rank_percent_summary_stats.csv')}")
    print(f"Saved: {os.path.join(out_dir, 'rank_percent_prediction_flow.csv')}")


if __name__ == "__main__":
    main()
