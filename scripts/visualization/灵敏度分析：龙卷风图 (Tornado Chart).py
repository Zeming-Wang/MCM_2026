import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_sensitivity_tornado(features, importance, uncertainty, output_path: str) -> str:
    features = list(features)
    importance = np.asarray(importance, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    order = np.argsort(-importance)
    features = [features[i] for i in order]
    importance = importance[order]
    uncertainty = uncertainty[order]

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=170)
    y_pos = np.arange(len(features))

    ax.barh(
        y_pos,
        importance,
        xerr=uncertainty,
        align="center",
        color="#A6CEE3",
        edgecolor="#5E60CE",
        capsize=4,
        alpha=0.9,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Global sensitivity (proxy, normalized)")
    ax.set_title("Sensitivity & Uncertainty (Bootstrap)")

    for i, v in enumerate(importance):
        ax.text(float(v) + 0.01, i, f"{float(v):.2f}", color="black", va="center")

    ax.grid(axis="x", linestyle="--", alpha=0.45)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")
    feature_table = os.path.join(out_dir, "sensitivity_feature_table.csv")

    if not os.path.exists(feature_table):
        script_path = os.path.join(
            project_root,
            "scripts",
            "visualization",
            "敏感性分析-贝叶斯后验区间与灵敏度雷达图 (Uncertainty Radar).py",
        )
        raise FileNotFoundError(f"Missing {feature_table}. Run: python \"{script_path}\"")

    df = pd.read_csv(feature_table, encoding="utf-8-sig")
    required = {"industry_idx", "age_z", "v_base", "judge_offset", "volatility", "y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in feature table: {missing}")

    def corr_imp(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = (x - float(np.mean(x))) / (float(np.std(x)) if float(np.std(x)) > 0 else 1.0)
        y = (y - float(np.mean(y))) / (float(np.std(y)) if float(np.std(y)) > 0 else 1.0)
        return abs(float(np.mean(x * y)))

    cats = ["Industry", "Age", "Fan Base", "Judge Offset", "Volatility"]
    x_cols = ["industry_idx", "age_z", "v_base", "judge_offset", "volatility"]
    base = np.asarray([corr_imp(df[c].to_numpy(), df["y"].to_numpy()) for c in x_cols], dtype=float)
    base = base / float(np.sum(base) if float(np.sum(base)) > 0 else 1.0)

    rng = np.random.default_rng(0)
    boots = []
    n = int(len(df))
    for _ in range(200):
        idx = rng.integers(0, n, size=n)
        d = df.iloc[idx]
        v = np.asarray([corr_imp(d[c].to_numpy(), d["y"].to_numpy()) for c in x_cols], dtype=float)
        v = v / float(np.sum(v) if float(np.sum(v)) > 0 else 1.0)
        boots.append(v)
    boots = np.asarray(boots, dtype=float)
    lo = np.quantile(boots, 0.025, axis=0)
    hi = np.quantile(boots, 0.975, axis=0)
    hw = (hi - lo) / 2.0

    out_path = os.path.join(out_dir, "tornado_chart.png")
    plot_sensitivity_tornado(cats, base, hw, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
