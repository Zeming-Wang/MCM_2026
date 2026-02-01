import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _standardize_pred(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Season"] = pd.to_numeric(out["Season"], errors="coerce").astype(int)
    out["Week"] = pd.to_numeric(out["Week"], errors="coerce").astype(int)
    out["Season_Type"] = out["Season_Type"].astype(str)
    out["Name"] = out["Name"].astype(str)
    for c in [
        "Judge_Points",
        "Judge_Rank",
        "Judge_Percent",
        "Fan_Vote_Percent",
        "Fan_Vote_Rank",
        "Predicted_Eliminated",
        "Actual_Eliminated",
        "V_Base",
        "Delta_V",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _extract_predicted_elim(df_pred: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df_pred.copy()
    df = df[df["Predicted_Eliminated"].astype(int) == 1]
    df = df.sort_values(["Season", "Week", "Name"])
    df = df.groupby(["Season", "Week"], as_index=False).first()
    df = df[["Season", "Week", "Name"]].rename(columns={"Name": label})
    return df


def build_compare_table(df_revote: pd.DataFrame, df_norevote: pd.DataFrame, df_base: pd.DataFrame) -> pd.DataFrame:
    a = _extract_predicted_elim(df_base, "Elim_Base")
    b = _extract_predicted_elim(df_norevote, "Elim_NoRevote")
    c = _extract_predicted_elim(df_revote, "Elim_Revote")
    out = b.merge(c, on=["Season", "Week"], how="inner").merge(a, on=["Season", "Week"], how="left")
    out["Diff_Revote_vs_NoRevote"] = (out["Elim_Revote"] != out["Elim_NoRevote"]).astype(int)
    out["Diff_Revote_vs_Base"] = np.where(
        out["Elim_Base"].notna(),
        (out["Elim_Revote"] != out["Elim_Base"]).astype(int),
        np.nan,
    )
    return out.sort_values(["Season", "Week"]).reset_index(drop=True)


def plot_stage_bump_a1(
    df_norevote: pd.DataFrame,
    df_compare: pd.DataFrame,
    season: int,
    out_path: str,
    bottom_k: int = 10,
):
    sns.set_context("talk")
    plt.style.use("seaborn-v0_8-paper")

    sub = df_norevote[df_norevote["Season"] == int(season)].copy()
    if sub.empty:
        raise ValueError(f"No rows for Season={season} in no-revote predictions.")

    cand = set()
    for (_, week), g in sub.groupby(["Season", "Week"], sort=True):
        g = g.sort_values("Fan_Vote_Rank", ascending=False).head(int(bottom_k))
        cand.update(g["Name"].astype(str).tolist())

    sub = sub[sub["Name"].isin(list(cand))].copy()
    if sub.empty:
        raise ValueError("Candidate subset is empty after bottom-K filtering.")

    fan = (
        sub.pivot_table(index="Week", columns="Name", values="Fan_Vote_Rank", aggfunc="mean")
        .sort_index()
    )
    judge = (
        sub.pivot_table(index="Week", columns="Name", values="Judge_Rank", aggfunc="mean")
        .sort_index()
    )

    names = [c for c in fan.columns.tolist() if c in judge.columns]
    if not names:
        raise ValueError("No overlapping candidate names between Fan and Judge ranks.")

    max_rank = int(max(float(np.nanmax(fan[n].to_numpy())) for n in names if fan[n].notna().any()))

    palette = sns.color_palette("husl", n_colors=len(names))
    color_map = {n: palette[i] for i, n in enumerate(names)}

    bg = "#0B1020"
    fg = "#E9EEF5"
    grid = "#7A8899"
    fig, ax = plt.subplots(figsize=(14.2, 7.8), dpi=190)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    dx = 0.11
    dy = 0.34

    weeks = sorted(fan.index.tolist())

    for n in names:
        y_f = fan[n]
        y_j = judge[n]
        if y_f.notna().sum() <= 1 and y_j.notna().sum() <= 1:
            continue

        xs_f = y_f.index.to_numpy(dtype=float)
        ys_f = y_f.to_numpy(dtype=float)
        mask_f = np.isfinite(ys_f)

        xs_j = y_j.index.to_numpy(dtype=float) + dx
        ys_j = y_j.to_numpy(dtype=float) + dy
        mask_j = np.isfinite(ys_j)

        c = color_map[n]

        ax.plot(
            xs_f[mask_f],
            ys_f[mask_f],
            linewidth=6.0,
            color=(0, 0, 0, 0.25),
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            xs_f[mask_f],
            ys_f[mask_f],
            linewidth=3.4,
            color=c,
            alpha=0.96,
            marker="o",
            markersize=4.8,
            solid_capstyle="round",
            zorder=3,
        )

        ax.plot(
            xs_j[mask_j],
            ys_j[mask_j],
            linewidth=4.5,
            color=(0, 0, 0, 0.18),
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            xs_j[mask_j],
            ys_j[mask_j],
            linewidth=2.3,
            color=c,
            alpha=0.52,
            marker="s",
            markersize=4.2,
            solid_capstyle="round",
            zorder=2,
        )

        for w in weeks:
            if w not in y_f.index or w not in y_j.index:
                continue
            yf = float(y_f.loc[w]) if pd.notna(y_f.loc[w]) else np.nan
            yj = float(y_j.loc[w]) if pd.notna(y_j.loc[w]) else np.nan
            if not (np.isfinite(yf) and np.isfinite(yj)):
                continue
            ax.plot(
                [float(w), float(w) + dx],
                [yf, yj + dy],
                color=grid,
                alpha=0.22,
                linewidth=1.0,
                zorder=0,
            )

    comp_s = df_compare[df_compare["Season"] == int(season)].copy()
    comp_s = comp_s.sort_values("Week")

    for _, row in comp_s.iterrows():
        w = int(row["Week"])
        name_nr = row.get("Elim_NoRevote")
        name_rv = row.get("Elim_Revote")
        if pd.notna(name_nr) and str(name_nr) in fan.columns and w in fan.index:
            yf = float(fan.loc[w, str(name_nr)])
            ax.scatter(
                [w],
                [yf],
                s=120,
                marker="X",
                color="#FFB703",
                edgecolors="#000000",
                linewidths=0.8,
                zorder=6,
            )
        if pd.notna(name_rv) and str(name_rv) in fan.columns and w in fan.index:
            yf = float(fan.loc[w, str(name_rv)])
            ax.scatter(
                [w],
                [yf],
                s=150,
                marker="*",
                color="#00F5D4",
                edgecolors="#000000",
                linewidths=0.8,
                zorder=7,
            )

    ax.set_title(
        f"Stage Bump (2.5D): Fan vs Judge Rank Trajectories — Season {season}",
        color=fg,
        pad=14,
        fontsize=16,
    )
    ax.set_xlabel("Week", color=fg)
    ax.set_ylabel("Rank (1 = best)", color=fg)
    ax.set_xlim(min(weeks) - 0.4, max(weeks) + 0.65)
    ax.set_ylim(1.0, max_rank + 0.75)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.18, color=grid)

    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.18))

    ax.tick_params(colors=fg)

    handles = [
        plt.Line2D([0], [0], color=fg, linewidth=3.2, marker="o", markersize=6, label="Fan rank layer"),
        plt.Line2D([0], [0], color=fg, linewidth=2.0, alpha=0.55, marker="s", markersize=6, label="Judge rank layer"),
        plt.Line2D([0], [0], color="#FFB703", marker="X", linestyle="None", markersize=9, label="Eliminated (No-Revote)"),
        plt.Line2D([0], [0], color="#00F5D4", marker="*", linestyle="None", markersize=12, label="Eliminated (Revote)"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor=bg,
        edgecolor=(1, 1, 1, 0.14),
        fontsize=10.5,
    )
    for text in leg.get_texts():
        text.set_color(fg)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_bottom2_geometry_table(df_norevote: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (season, week), g in df_norevote.groupby(["Season", "Week"], sort=True):
        if len(g) < 2:
            continue
        g = g.sort_values("Fan_Vote_Percent", ascending=True)
        a = g.iloc[0]
        b = g.iloc[1]
        fan_gap = float(b["Fan_Vote_Percent"]) - float(a["Fan_Vote_Percent"])
        judge_gap = float(b["Judge_Points"]) - float(a["Judge_Points"])
        out_rows.append(
            {
                "Season": int(season),
                "Week": int(week),
                "Season_Type": str(a["Season_Type"]),
                "N_Active": int(len(g)),
                "Bottom1_Name": str(a["Name"]),
                "Bottom2_Name": str(b["Name"]),
                "Fan_Gap_Bottom2": float(fan_gap),
                "Judge_Gap_Bottom2": float(judge_gap),
            }
        )
    return pd.DataFrame(out_rows)


def plot_bottom2_geometry_b(
    df_geom: pd.DataFrame,
    out_path: str,
    annotate_top: int = 8,
):
    sns.set_context("talk")
    plt.style.use("seaborn-v0_8-paper")

    bg = "#0B1020"
    fg = "#E9EEF5"
    grid = "#7A8899"

    fig, ax = plt.subplots(figsize=(12.8, 7.4), dpi=190)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    df = df_geom.copy()
    df["Changed"] = df["Changed"].astype(int)

    palette = {0: "#8D99AE", 1: "#FF4D6D"}

    sns.scatterplot(
        data=df,
        x="Fan_Gap_Bottom2",
        y="Judge_Gap_Bottom2",
        hue="Changed",
        style="Season_Type",
        size="N_Active",
        sizes=(40, 220),
        alpha=0.88,
        palette=palette,
        ax=ax,
        edgecolor=(0, 0, 0, 0.25),
        linewidth=0.6,
    )

    try:
        sns.kdeplot(
            data=df,
            x="Fan_Gap_Bottom2",
            y="Judge_Gap_Bottom2",
            levels=6,
            color=fg,
            linewidths=1.0,
            alpha=0.22,
            ax=ax,
        )
    except Exception:
        pass

    ax.axhline(0.0, color=fg, linestyle="--", linewidth=1.2, alpha=0.35)
    ax.axvline(0.0, color=fg, linestyle="--", linewidth=1.2, alpha=0.25)

    ax.set_title("Bottom-2 Geometry: Margin Space Where Revote Flips Elimination", color=fg, pad=14, fontsize=16)
    ax.set_xlabel("Fan bottom-2 margin (2nd worst − worst)", color=fg)
    ax.set_ylabel("Judge bottom-2 margin on the same two (2nd − worst)", color=fg)
    ax.grid(True, linestyle="--", alpha=0.18, color=grid)

    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.18))
    ax.tick_params(colors=fg)

    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        facecolor=bg,
        edgecolor=(1, 1, 1, 0.14),
        fontsize=10.0,
    )
    for text in leg.get_texts():
        text.set_color(fg)

    changed = df[df["Changed"] == 1].copy()
    if not changed.empty:
        changed = changed.sort_values(["Fan_Gap_Bottom2", "N_Active"], ascending=[True, False]).head(int(annotate_top))
        for _, r in changed.iterrows():
            ax.text(
                float(r["Fan_Gap_Bottom2"]),
                float(r["Judge_Gap_Bottom2"]),
                f"S{int(r['Season'])}W{int(r['Week'])}",
                color=fg,
                fontsize=9.4,
                alpha=0.9,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _mosaic_pngs(image_paths: list[str], out_path: str, ncols: int, title: str) -> str:
    valid = [p for p in image_paths if isinstance(p, str) and os.path.exists(p)]
    if not valid:
        raise ValueError("No images found for mosaic.")

    imgs = [plt.imread(p) for p in valid]
    n = len(imgs)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    bg = "#0B1020"
    fg = "#E9EEF5"

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.4 * ncols, 3.2 * nrows), dpi=190)
    fig.patch.set_facecolor(bg)
    if isinstance(axes, np.ndarray):
        ax_list = axes.ravel().tolist()
    else:
        ax_list = [axes]

    for ax in ax_list:
        ax.set_facecolor(bg)
        ax.axis("off")

    for i, (ax, im) in enumerate(zip(ax_list, imgs)):
        ax.imshow(im)
        ax.set_title(os.path.basename(valid[i]).replace(".png", ""), color=fg, fontsize=9)

    fig.suptitle(title, color=fg, fontsize=16, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    root = _project_root()
    pred_base = os.path.join(root, "data", "processed", "model1_fan_vote_predictions_enriched.csv")
    pred_norevote = os.path.join(root, "data", "processed", "model1_fan_vote_predictions_no_revote.csv")
    pred_revote = os.path.join(root, "data", "processed", "model1_fan_vote_predictions_with_revote_all.csv")

    df_base = _standardize_pred(_load_csv(pred_base))
    df_nr = _standardize_pred(_load_csv(pred_norevote))
    df_rv = _standardize_pred(_load_csv(pred_revote))

    selected_seasons = [2, 4, 11, 27, 28, 29, 34]
    max_weeks = 7

    df_base = df_base[df_base["Season"].isin(selected_seasons) & (df_base["Week"] <= int(max_weeks))].copy()
    df_nr = df_nr[df_nr["Season"].isin(selected_seasons) & (df_nr["Week"] <= int(max_weeks))].copy()
    df_rv = df_rv[df_rv["Season"].isin(selected_seasons) & (df_rv["Week"] <= int(max_weeks))].copy()

    out_dir = os.path.join(root, "scripts", "visualization", "outputs")
    run_dir = os.path.join(out_dir, f"revote_compare_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    df_cmp = build_compare_table(df_rv, df_nr, df_base)
    cmp_path = os.path.join(run_dir, "revote_compare_table.csv")
    df_cmp.to_csv(cmp_path, index=False, encoding="utf-8-sig")

    diff_rate = float(df_cmp["Diff_Revote_vs_NoRevote"].mean()) if not df_cmp.empty else 0.0
    per_season = (
        df_cmp.groupby("Season", as_index=False)["Diff_Revote_vs_NoRevote"]
        .mean()
        .rename(columns={"Diff_Revote_vs_NoRevote": "Flip_Rate"})
        .sort_values("Flip_Rate", ascending=False)
    )
    per_season_path = os.path.join(run_dir, "revote_flip_rate_by_season.csv")
    per_season.to_csv(per_season_path, index=False, encoding="utf-8-sig")

    a1_paths = []
    for season in selected_seasons:
        if int(season) not in set(df_nr["Season"].unique().tolist()):
            continue
        out_a1 = os.path.join(run_dir, f"plot_A1_stage_bump_season_{int(season)}.png")
        try:
            plot_stage_bump_a1(df_nr, df_cmp, season=int(season), out_path=out_a1, bottom_k=10)
            a1_paths.append(out_a1)
        except Exception:
            pass

    if a1_paths:
        out_a1_grid = os.path.join(run_dir, "plot_A1_stage_bump_grid.png")
        _mosaic_pngs(
            a1_paths,
            out_path=out_a1_grid,
            ncols=4,
            title="A1 Stage Bump Grid (Fan vs Judge, 2.5D) — Selected Seasons, First 7 Weeks",
        )

    df_geom = build_bottom2_geometry_table(df_nr)
    df_geom = df_geom.merge(
        df_cmp[["Season", "Week", "Diff_Revote_vs_NoRevote"]],
        on=["Season", "Week"],
        how="left",
    ).rename(columns={"Diff_Revote_vs_NoRevote": "Changed"})
    df_geom["Changed"] = pd.to_numeric(df_geom["Changed"], errors="coerce").fillna(0).astype(int)
    geom_path = os.path.join(run_dir, "bottom2_geometry_table.csv")
    df_geom.to_csv(geom_path, index=False, encoding="utf-8-sig")

    out_b = os.path.join(run_dir, "plot_B_bottom2_geometry.png")
    plot_bottom2_geometry_b(df_geom, out_path=out_b, annotate_top=10)

    b_paths = []
    for season in selected_seasons:
        g = df_geom[df_geom["Season"] == int(season)].copy()
        if g.empty:
            continue
        out_bs = os.path.join(run_dir, f"plot_B_bottom2_geometry_season_{int(season)}.png")
        try:
            plot_bottom2_geometry_b(g, out_path=out_bs, annotate_top=6)
            b_paths.append(out_bs)
        except Exception:
            pass

    if b_paths:
        out_b_grid = os.path.join(run_dir, "plot_B_bottom2_geometry_grid.png")
        _mosaic_pngs(
            b_paths,
            out_path=out_b_grid,
            ncols=4,
            title="B Bottom-2 Geometry Grid — Selected Seasons, First 7 Weeks",
        )

    print(f"Saved: {cmp_path} (rows={len(df_cmp)}, flip_rate={diff_rate:.3f})")
    print(f"Saved: {per_season_path} (rows={len(per_season)})")
    print(f"Saved: {geom_path} (rows={len(df_geom)})")
    if a1_paths:
        print(f"Saved A1 season plots: {len(a1_paths)}")
    if b_paths:
        print(f"Saved B season plots: {len(b_paths)}")
    print(f"Saved: {out_b}")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()