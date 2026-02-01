import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    exp_x = np.exp(x - float(np.max(x)))
    s = float(np.sum(exp_x))
    return exp_x / (s if s > 0 else 1.0)


def _prepare_predictions(df_pred: pd.DataFrame) -> pd.DataFrame:
    df = df_pred.copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype(int)
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype(int)
    df["Name"] = df["Name"].astype(str)
    df["Fan_Vote_Percent"] = pd.to_numeric(df.get("Fan_Vote_Percent"), errors="coerce").fillna(0.0)
    df["V_Base"] = pd.to_numeric(df.get("V_Base"), errors="coerce").fillna(0.0)
    df["Delta_V"] = pd.to_numeric(df.get("Delta_V"), errors="coerce").fillna(0.0)
    return df


def _safe_sigma(v_base: np.ndarray, delta_v: np.ndarray) -> float:
    sigma = float(np.std(delta_v))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(v_base))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1e-6
    return sigma


def compute_elimination_risk(
    df_pred: pd.DataFrame,
    alpha: float = 0.5,
    draws: int = 800,
    seed: int = 0,
) -> pd.DataFrame:
    df = _prepare_predictions(df_pred)
    rng = np.random.default_rng(int(seed))
    out_rows = []

    for (season, week), g in df.groupby(["Season", "Week"], sort=True):
        if len(g) <= 1:
            continue
        names = g["Name"].astype(str).to_numpy()
        v_base = g["V_Base"].to_numpy(dtype=float)
        delta_v = g["Delta_V"].to_numpy(dtype=float)

        point = g["Fan_Vote_Percent"].to_numpy(dtype=float)
        if float(np.sum(point)) <= 0:
            point = _softmax(v_base + alpha * delta_v)
        else:
            point = point / float(np.sum(point))

        sigma = _safe_sigma(v_base, delta_v)
        n = len(g)
        fan_samples = np.zeros((int(draws), n), dtype=float)

        for i in range(int(draws)):
            eps = rng.normal(0.0, sigma, size=n)
            raw = v_base + alpha * (delta_v + eps)
            fan_samples[i] = _softmax(raw)

        elim_idx = np.argmin(fan_samples, axis=1)
        elim_counts = np.bincount(elim_idx, minlength=n).astype(float) / float(draws)

        bottom2_counts = np.zeros(n, dtype=float)
        k = 2 if n >= 2 else n
        for i in range(int(draws)):
            idx = np.argpartition(fan_samples[i], k - 1)[:k]
            bottom2_counts[idx] += 1.0
        bottom2_counts = bottom2_counts / float(draws)

        p05 = np.quantile(fan_samples, 0.05, axis=0)
        p50 = np.quantile(fan_samples, 0.50, axis=0)
        p95 = np.quantile(fan_samples, 0.95, axis=0)

        for i in range(n):
            out_rows.append(
                {
                    "Season": int(season),
                    "Week": int(week),
                    "Name": str(names[i]),
                    "Elimination_Prob": float(elim_counts[i]),
                    "Bottom2_Prob": float(bottom2_counts[i]),
                    "Median_FanVote": float(p50[i]),
                    "CI05_FanVote": float(p05[i]),
                    "CI95_FanVote": float(p95[i]),
                    "Point_FanVote": float(point[i]),
                    "V_Base": float(v_base[i]),
                    "Delta_V": float(delta_v[i]),
                }
            )

    return pd.DataFrame(out_rows)


def plot_risk_curves(df_risk: pd.DataFrame, out_dir: str, top_k: int = 6) -> list[str]:
    saved = []
    plt.style.use("seaborn-v0_8-paper")
    for season in sorted(df_risk["Season"].unique().tolist()):
        sub = df_risk[df_risk["Season"] == int(season)].copy()
        if sub.empty:
            continue
        pivot = (
            sub.pivot_table(index="Week", columns="Name", values="Elimination_Prob", aggfunc="mean")
            .fillna(0.0)
        )
        avg = pivot.mean(axis=0).sort_values(ascending=False)
        names = avg.head(int(top_k)).index.tolist()
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=170)
        for name in names:
            ax.plot(pivot.index, pivot[name], marker="o", linewidth=2.0, label=str(name))
        ax.set_title(f"Elimination Risk Curves (Season {season})")
        ax.set_xlabel("Week")
        ax.set_ylabel("Elimination Probability")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"elimination_risk_curve_season_{season}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
    return saved


def plot_weekwise_risk_bar(df_risk: pd.DataFrame, out_dir: str, top_k: int = 6) -> list[str]:
    saved = []
    plt.style.use("seaborn-v0_8-paper")
    for (season, week), g in df_risk.groupby(["Season", "Week"], sort=True):
        g = g.sort_values("Elimination_Prob", ascending=False).head(int(top_k))
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=170)
        ax.barh(g["Name"], g["Elimination_Prob"], color="#E76F51", alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(f"Top-{len(g)} Elimination Risk (Season {season} Week {week})")
        ax.set_xlabel("Elimination Probability")
        ax.set_xlim(0.0, 1.0)
        for i, row in g.reset_index(drop=True).iterrows():
            ax.text(
                float(row["Elimination_Prob"]) + 0.01,
                i,
                f"Point={row['Point_FanVote']:.3f}",
                va="center",
                fontsize=9,
            )
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"weekwise_risk_bar_season_{season}_week_{week}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
    return saved


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pred_path = os.path.join(project_root, "data", "processed", "model1_fan_vote_predictions.csv")
    pred_path_fallback = os.path.join(project_root, "results", "model1_fan_vote_predictions_subset.csv")
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")

    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    elif os.path.exists(pred_path_fallback):
        df_pred = pd.read_csv(pred_path_fallback, encoding="utf-8-sig")
    else:
        raise FileNotFoundError(
            "Missing predictions CSV. Run src/models/model1_prediction_newest_canrun.py to generate "
            f"{pred_path} (or run subset script to generate {pred_path_fallback})."
        )

    os.makedirs(out_dir, exist_ok=True)

    df_risk = compute_elimination_risk(df_pred, alpha=0.5, draws=800, seed=0)
    out_csv = os.path.join(out_dir, "elimination_risk_table.csv")
    df_risk.to_csv(out_csv, index=False, encoding="utf-8-sig")

    risk_curves = plot_risk_curves(df_risk, out_dir=out_dir, top_k=6)
    week_bars = plot_weekwise_risk_bar(df_risk, out_dir=out_dir, top_k=6)

    print(f"Saved: {out_csv} (rows={len(df_risk)})")
    if risk_curves:
        print(f"Saved: {risk_curves[0]} (and {len(risk_curves) - 1} more)")
    if week_bars:
        print(f"Saved: {week_bars[0]} (and {len(week_bars) - 1} more)")


if __name__ == "__main__":
    main()
