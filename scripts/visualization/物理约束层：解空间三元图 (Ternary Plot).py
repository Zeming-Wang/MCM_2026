import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def _barycentric_to_cart(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    x = p[:, 1] + 0.5 * p[:, 2]
    y = (np.sqrt(3) / 2.0) * p[:, 2]
    return x, y


def _plot_ternary_axes(ax):
    ax.plot([0, 1], [0, 0], color="#1F2937", linewidth=1.2)
    ax.plot([0, 0.5], [0, np.sqrt(3) / 2.0], color="#1F2937", linewidth=1.2)
    ax.plot([1, 0.5], [0, np.sqrt(3) / 2.0], color="#1F2937", linewidth=1.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3) / 2.0 + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def plot_ternary_feasible_region(p_judge_3: np.ndarray, eliminated_idx: int, labels: list[str], output_path: str) -> str:
    p_judge_3 = np.asarray(p_judge_3, dtype=float).reshape(3)
    p_judge_3 = p_judge_3 / float(np.sum(p_judge_3) if float(np.sum(p_judge_3)) > 0 else 1.0)
    e = int(eliminated_idx)

    rng = np.random.default_rng(0)
    points = rng.dirichlet(np.ones(3), size=25000)

    feasible = []
    for p_fan in points:
        combined = p_judge_3 + p_fan
        ok = True
        for i in range(3):
            if i == e:
                continue
            if combined[e] > combined[i] + 1e-12:
                ok = False
                break
        if ok:
            feasible.append(p_fan)
    feasible = np.asarray(feasible, dtype=float)

    x_all, y_all = _barycentric_to_cart(points)
    x_f, y_f = _barycentric_to_cart(feasible) if len(feasible) else (np.asarray([]), np.asarray([]))

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=170)
    _plot_ternary_axes(ax)
    ax.scatter(x_all, y_all, s=2, color="#9CA3AF", alpha=0.10, linewidths=0)
    if len(feasible):
        ax.scatter(x_f, y_f, s=2, color="#2563EB", alpha=0.32, linewidths=0)

    ax.text(-0.02, -0.03, labels[0], ha="left", va="top", fontsize=10)
    ax.text(1.02, -0.03, labels[1], ha="right", va="top", fontsize=10)
    ax.text(0.5, np.sqrt(3) / 2.0 + 0.03, labels[2], ha="center", va="bottom", fontsize=10)
    ax.set_title("Feasible Fan-Vote Region on 3-Simplex (Percent System)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_path = os.path.join(project_root, "data", "processed", "processed_mcm_wide_clean.csv")
    raw_path = os.path.join(project_root, "data", "raw", "2026_MCM_Problem_C_Data.csv")
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")

    df_wide = _ensure_processed_wide(processed_path, raw_path)
    seasons = sorted(pd.to_numeric(df_wide["season"], errors="coerce").dropna().astype(int).unique().tolist())

    chosen = None
    for season in seasons:
        if season <= 2 or season >= 28:
            continue
        season_df = df_wide[df_wide["season"].astype(int) == int(season)].copy().reset_index(drop=True)
        for week in range(1, 12):
            judge_cols = [c for c in _week_judge_cols(week) if c in season_df.columns]
            if not judge_cols:
                continue
            w = season_df[["celebrity_name", "results"] + judge_cols].copy()
            for c in judge_cols:
                w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0)
            pts = w[judge_cols].sum(axis=1).to_numpy()
            active = pts > 0
            w = w[active].reset_index(drop=True)
            pts = pts[active]
            if len(w) < 3:
                continue
            elim_flag = w["results"].astype(str).str.contains(f"Eliminated Week {week}", case=False, na=False)
            if not bool(elim_flag.any()):
                continue
            elim_name = str(w.loc[elim_flag, "celebrity_name"].iloc[0])
            s = float(np.sum(pts))
            p_j = pts / (s if s > 0 else 1.0)
            idx = np.argsort(p_j)[:3]
            names3 = w.loc[idx, "celebrity_name"].astype(str).tolist()
            if elim_name not in names3:
                continue
            e = int(names3.index(elim_name))
            chosen = (season, week, p_j[idx], e, names3)
            break
        if chosen is not None:
            break

    if chosen is None:
        raise RuntimeError("No suitable percent-system season-week found for ternary visualization.")

    season, week, p_j3, e, names3 = chosen
    out_path = os.path.join(out_dir, f"ternary_feasible_region_S{season}_W{week}.png")
    plot_ternary_feasible_region(p_j3, e, names3, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
