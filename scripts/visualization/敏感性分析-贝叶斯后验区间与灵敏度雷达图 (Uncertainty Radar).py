import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return (x - mu) / (sd if sd > 0 else 1.0)


def _safe_corr_importance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        return 0.0
    x = _zscore(x)
    y = _zscore(y)
    v = float(np.mean(x * y))
    return abs(v)


def estimate_global_sensitivity(df: pd.DataFrame, bootstrap: int = 200, seed: int = 0):
    rng = np.random.default_rng(int(seed))
    categories = ["Industry", "Age", "Fan Base", "Judge Offset", "Volatility"]

    base = np.asarray(
        [
            _safe_corr_importance(df["industry_idx"].to_numpy(), df["y"].to_numpy()),
            _safe_corr_importance(df["age_z"].to_numpy(), df["y"].to_numpy()),
            _safe_corr_importance(df["v_base"].to_numpy(), df["y"].to_numpy()),
            _safe_corr_importance(df["judge_offset"].to_numpy(), df["y"].to_numpy()),
            _safe_corr_importance(df["volatility"].to_numpy(), df["y"].to_numpy()),
        ],
        dtype=float,
    )
    base = base / float(np.sum(base) if float(np.sum(base)) > 0 else 1.0)

    boots = []
    n = int(len(df))
    for _ in range(int(bootstrap)):
        idx = rng.integers(0, n, size=n)
        d = df.iloc[idx]
        v = np.asarray(
            [
                _safe_corr_importance(d["industry_idx"].to_numpy(), d["y"].to_numpy()),
                _safe_corr_importance(d["age_z"].to_numpy(), d["y"].to_numpy()),
                _safe_corr_importance(d["v_base"].to_numpy(), d["y"].to_numpy()),
                _safe_corr_importance(d["judge_offset"].to_numpy(), d["y"].to_numpy()),
                _safe_corr_importance(d["volatility"].to_numpy(), d["y"].to_numpy()),
            ],
            dtype=float,
        )
        v = v / float(np.sum(v) if float(np.sum(v)) > 0 else 1.0)
        boots.append(v)
    boots = np.asarray(boots, dtype=float)
    lo = np.quantile(boots, 0.025, axis=0)
    hi = np.quantile(boots, 0.975, axis=0)
    hdi_half_width = (hi - lo) / 2.0

    return categories, base, hdi_half_width


def plot_sensitivity_radar(categories, sobol_like, hdi_half_width, output_path: str) -> str:
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    values = list(sobol_like) + [float(sobol_like[0])]
    error = list(hdi_half_width) + [float(hdi_half_width[0])]

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(8.6, 8.2), subplot_kw=dict(polar=True), dpi=170)

    ax.plot(angles, values, color="#7A70B5", linewidth=2.4, label="Global sensitivity (proxy)")
    ax.fill(angles, values, color="#839DD1", alpha=0.22)

    upper = [v + e for v, e in zip(values, error)]
    lower = [max(v - e, 0.0) for v, e in zip(values, error)]
    ax.fill_between(angles, lower, upper, color="#F1766D", alpha=0.12, label="Bootstrap 95% interval")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, max(0.6, float(max(upper)) * 1.05))
    ax.set_title("Global Sensitivity Radar & Uncertainty", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=True)

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
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")

    if not os.path.exists(processed_path):
        import sys

        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from scripts.preprocessing.process_data import clean_raw_to_wide_clean

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_raw = pd.read_csv(raw_path, encoding="utf-8-sig")
        df_wide = clean_raw_to_wide_clean(df_raw)
        df_wide.to_csv(processed_path, index=False, encoding="utf-8-sig")
    df_wide = pd.read_csv(processed_path, encoding="utf-8-sig")

    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    elif os.path.exists(pred_path_fallback):
        df_pred = pd.read_csv(pred_path_fallback, encoding="utf-8-sig")
    else:
        raise FileNotFoundError(
            "Missing predictions CSV. Run src/models/model1_prediction_newest_canrun.py to generate "
            f"{pred_path} (or run subset script to generate {pred_path_fallback})."
        )

    df_pred["Season"] = pd.to_numeric(df_pred["Season"], errors="coerce").astype(int)
    df_pred["Week"] = pd.to_numeric(df_pred["Week"], errors="coerce").astype(int)
    df_pred["Name"] = df_pred["Name"].astype(str)
    df_pred["Fan_Vote_Percent"] = pd.to_numeric(df_pred["Fan_Vote_Percent"], errors="coerce").fillna(0.0)
    df_pred["V_Base"] = pd.to_numeric(df_pred.get("V_Base"), errors="coerce").fillna(0.0)
    df_pred["Delta_V"] = pd.to_numeric(df_pred.get("Delta_V"), errors="coerce").fillna(0.0)

    age_map = {}
    ind_map = {}
    for _, row in df_wide.iterrows():
        age_map[(int(row["season"]), str(row["celebrity_name"]))] = row.get("celebrity_age_during_season", np.nan)
        ind_map[(int(row["season"]), str(row["celebrity_name"]))] = row.get("industry_idx", np.nan)

    rows = []
    for (season, week), g in df_pred.groupby(["Season", "Week"], sort=False):
        if len(g) <= 2:
            continue
        ages = np.asarray([age_map.get((int(season), str(n)), np.nan) for n in g["Name"].tolist()], dtype=float)
        ages = np.nan_to_num(ages, nan=float(np.nanmean(ages)) if np.isfinite(np.nanmean(ages)) else 0.0)
        age_z = _zscore(ages)
        industry = np.asarray([ind_map.get((int(season), str(n)), np.nan) for n in g["Name"].tolist()], dtype=float)
        industry = np.nan_to_num(industry, nan=float(np.nanmean(industry)) if np.isfinite(np.nanmean(industry)) else 0.0)
        industry_z = _zscore(industry)

        y = g["Fan_Vote_Percent"].to_numpy(dtype=float)
        y = np.clip(y, 1e-12, 1.0)
        y = np.log(y) - float(np.mean(np.log(y)))

        v_base = g["V_Base"].to_numpy(dtype=float)
        delta_v = g["Delta_V"].to_numpy(dtype=float)
        judge_offset = _zscore(v_base)
        volatility = np.abs(delta_v)

        for i in range(len(g)):
            rows.append(
                {
                    "industry_idx": float(industry_z[i]),
                    "age_z": float(age_z[i]),
                    "v_base": float(v_base[i]),
                    "judge_offset": float(judge_offset[i]),
                    "volatility": float(volatility[i]),
                    "y": float(y[i]),
                }
            )

    df_feat = pd.DataFrame(rows)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    df_feat.to_csv(os.path.join(out_dir, "sensitivity_feature_table.csv"), index=False, encoding="utf-8-sig")

    categories, sobol_like, hdi_hw = estimate_global_sensitivity(df_feat, bootstrap=200, seed=0)
    out_path = os.path.join(out_dir, "uncertainty_radar.png")
    plot_sensitivity_radar(categories, sobol_like, hdi_hw, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
