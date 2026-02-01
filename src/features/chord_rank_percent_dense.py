import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle


def _load_scoring(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")
    df["Name"] = df["Name"].astype(str)
    df["Predicted_Eliminated_Name"] = df["Predicted_Eliminated_Name"].astype(str)
    return df


def _build_flow(rank_score: pd.DataFrame, percent_score: pd.DataFrame) -> pd.DataFrame:
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


def _polar_to_cart(r: float, theta: float) -> tuple[float, float]:
    return float(r * np.cos(theta)), float(r * np.sin(theta))


def _prepare_flow(flow: pd.DataFrame, top_n: int, min_count: int) -> pd.DataFrame:
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
    flow = flow[flow["Count"] >= int(min_count)].reset_index(drop=True)
    return flow


def plot_chord_dense(flow: pd.DataFrame, out_path: str) -> str:
    names = sorted(set(flow["Rank_Predicted"].tolist() + flow["Percent_Predicted"].tolist()))
    n = len(names)
    if n == 0:
        raise RuntimeError("No flow records available for chord diagram.")
    counts_by_name = (
        pd.concat(
            [
                flow[["Rank_Predicted", "Count"]].rename(columns={"Rank_Predicted": "Name"}),
                flow[["Percent_Predicted", "Count"]].rename(columns={"Percent_Predicted": "Name"}),
            ],
            ignore_index=True,
        )
        .groupby("Name", dropna=False)["Count"]
        .sum()
    )
    names = sorted(names, key=lambda x: (-counts_by_name.get(x, 0.0), str(x)))
    n = len(names)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + np.pi / 2.0
    radius = 1.0
    inner = 0.82
    palette = sns.color_palette("Set2", n_colors=max(n, 3))
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(names)}
    flow = flow.copy()
    max_count = float(flow["Count"].max()) if not flow.empty else 1.0
    count_scaled = np.sqrt(flow["Count"].to_numpy(dtype=float) / max_count)
    flow["Width"] = 0.6 + 4.2 * count_scaled
    flow["Alpha"] = 0.2 + 0.65 * count_scaled

    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(11.6, 11.6), dpi=260)
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
            x * 1.1,
            y * 1.1,
            name,
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=8.6,
            color="#1f1f1f",
        )
        ax.plot([x * 0.97, x * 1.02], [y * 0.97, y * 1.02], color=color_map[name], lw=2.6)

    for _, row in flow.iterrows():
        a = row["Rank_Predicted"]
        b = row["Percent_Predicted"]
        if float(row["Count"]) <= 0:
            continue
        ia = names.index(a)
        ib = names.index(b)
        ta = angles[ia]
        tb = angles[ib]
        start = np.array(_polar_to_cart(inner, ta))
        end = np.array(_polar_to_cart(inner, tb))
        if a == b:
            mid = start * 0.62
            verts = [start, mid, end]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)
            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=color_map.get(a, "#666666"),
                linewidth=float(row["Width"]) + 0.6,
                alpha=min(0.95, float(row["Alpha"]) + 0.2),
            )
            ax.add_patch(patch)
            continue
        c1 = start * 0.3
        c2 = end * 0.3
        verts = [start, c1, c2, end]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=color_map.get(a, "#666666"),
            linewidth=float(row["Width"]),
            alpha=float(row["Alpha"]),
        )
        ax.add_patch(patch)

    ring = Circle((0.0, 0.0), radius=inner, fill=False, edgecolor="#D0D0D0", linewidth=1.1)
    ax.add_patch(ring)
    ax.set_title("Chord Diagram: Rank vs Percent Elimination Predictions (Dense)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data", "processed")
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    rank_score_path = os.path.join(data_dir, "model1_rank_sum_scoring.csv")
    percent_score_path = os.path.join(data_dir, "model1_percent_sum_scoring.csv")
    if not os.path.exists(rank_score_path):
        raise FileNotFoundError(f"Missing {rank_score_path}")
    if not os.path.exists(percent_score_path):
        raise FileNotFoundError(f"Missing {percent_score_path}")

    rank_score = _load_scoring(rank_score_path)
    percent_score = _load_scoring(percent_score_path)
    flow = _build_flow(rank_score, percent_score)
    flow = _prepare_flow(flow, top_n=24, min_count=1)
    flow.to_csv(os.path.join(out_dir, "rank_percent_prediction_flow_dense.csv"), index=False, encoding="utf-8-sig")

    out_path = os.path.join(out_dir, "chord_rank_vs_percent_predictions_dense.png")
    plot_chord_dense(flow, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
