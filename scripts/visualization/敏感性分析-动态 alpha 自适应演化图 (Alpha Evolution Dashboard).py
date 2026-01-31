import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p / float(np.sum(p) if float(np.sum(p)) > 0 else 1.0)
    p = np.clip(p, 1e-15, 1.0)
    return float(-np.sum(p * np.log(p)))


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
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


def plot_alpha_evolution(weeks, alpha_vals, entropy_vals, output_path):
    plt.style.use("seaborn-v0_8-paper")
    fig, ax1 = plt.subplots(figsize=(11.0, 5.6), dpi=170)

    color1 = "#899FB0"
    ax1.set_xlabel("Competition Week")
    ax1.set_ylabel("Residual Influence (proxy)", color=color1, fontsize=11)
    ax1.plot(
        weeks,
        alpha_vals,
        marker="o",
        color=color1,
        linewidth=2.6,
        label="Residual influence",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#81B3A9"
    ax2.set_ylabel("Normalized entropy of feasible prior", color=color2, fontsize=11)
    ax2.plot(
        weeks,
        entropy_vals,
        marker="s",
        linestyle="--",
        color=color2,
        alpha=0.85,
        linewidth=2.2,
        label="Entropy",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Dynamic Balancing: Constraint Feasibility vs Residual Influence")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pred_path = os.path.join(project_root, "data", "processed", "model1_fan_vote_predictions.csv")
    pred_path_fallback = os.path.join(project_root, "results", "model1_fan_vote_predictions_subset.csv")
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")

    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path, encoding="utf-8-sig")
    elif os.path.exists(pred_path_fallback):
        df = pd.read_csv(pred_path_fallback, encoding="utf-8-sig")
    else:
        raise FileNotFoundError(
            "Missing predictions CSV. Run src/models/model1_prediction_newest_canrun.py to generate "
            f"{pred_path} (or run subset script to generate {pred_path_fallback})."
        )

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype(int)
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype(int)
    df["V_Base"] = pd.to_numeric(df.get("V_Base"), errors="coerce").fillna(0.0)
    df["Delta_V"] = pd.to_numeric(df.get("Delta_V"), errors="coerce").fillna(0.0)

    week_rows = []
    for week, g in df.groupby("Week", sort=True):
        alpha = 0.5
        ratios = []
        ent = []
        for (season, _week), sw in g.groupby(["Season", "Week"], sort=False):
            vb = sw["V_Base"].to_numpy(dtype=float)
            dv = sw["Delta_V"].to_numpy(dtype=float)
            if vb.size <= 1:
                continue
            vb = _project_to_simplex(vb)
            ent.append(_entropy(vb) / float(np.log(len(vb))))
            num = float(np.linalg.norm(alpha * dv))
            den = float(np.linalg.norm(vb)) + 1e-12
            ratios.append(num / den)
        if ratios:
            influence = float(np.mean(ratios))
            influence = influence / (1.0 + influence)
            week_rows.append({"Week": int(week), "Influence": influence, "Entropy": float(np.mean(ent))})

    df_week = pd.DataFrame(week_rows).sort_values("Week")
    df_week.to_csv(os.path.join(out_dir, "alpha_entropy_weekly.csv"), index=False, encoding="utf-8-sig")

    out_path = os.path.join(out_dir, "alpha_evolution.png")
    plot_alpha_evolution(
        df_week["Week"].tolist(),
        df_week["Influence"].tolist(),
        df_week["Entropy"].tolist(),
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
