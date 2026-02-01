import os
import re
import unicodedata
import logging
import warnings

import numpy as np
import pandas as pd
import pymc as pm


def normalize_name(x) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’", "'").replace("`", "'")
    return s


def get_scoring_system(season: int) -> str:
    if season <= 2 or season >= 28:
        return "rank"
    return "percent"


def week_judge_cols(week: int) -> list[str]:
    return [
        f"week{week}_judge1_score",
        f"week{week}_judge2_score",
        f"week{week}_judge3_score",
        f"week{week}_judge4_score",
    ]


def rank_descending(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.argsort(np.argsort(-values)) + 1


class RankPhysicalConstraintModel:
    def __init__(self, judge_ranks, eliminated_idx=None):
        self.judge_ranks = np.asarray(judge_ranks)
        self.eliminated_idx = eliminated_idx
        self.n = len(judge_ranks)

    def get_feasible_fan_rank_prior(self):
        if self.eliminated_idx is None:
            return np.ones(self.n) / (self.n if self.n > 0 else 1.0)

        n = int(self.n)
        e = int(self.eliminated_idx)
        if n <= 0:
            return np.ones(1)

        rng = np.random.default_rng(0)
        accepted = []
        judge_ranks = np.asarray(self.judge_ranks).astype(int)

        max_accept = min(500, 60 * n)
        max_draws = 30000

        for _ in range(max_draws):
            latent = rng.normal(0.0, 1.0, size=n)
            fan_ranks = np.argsort(np.argsort(-latent)) + 1
            rank_sum = judge_ranks + fan_ranks

            if np.all(rank_sum[e] >= np.delete(rank_sum, e)):
                exp_v = np.exp(latent - float(np.max(latent)))
                denom = float(np.sum(exp_v))
                p = exp_v / (denom if denom > 0 else 1.0)
                accepted.append(p)
                if len(accepted) >= max_accept:
                    break

        if accepted:
            v = np.mean(np.asarray(accepted), axis=0)
            s = float(np.sum(v))
            return v / (s if s > 0 else 1.0)

        v_base = np.ones(n)
        v_base[e] = 0.3
        return v_base / float(np.sum(v_base))


class PercentPhysicalConstraintModel:
    def __init__(self, judge_percent, eliminated_idx=None):
        self.judge_percent = np.asarray(judge_percent)
        self.eliminated_idx = eliminated_idx
        self.n = len(judge_percent)

    def get_feasible_fan_percent_prior(self):
        if self.eliminated_idx is None:
            return np.ones(self.n) / (self.n if self.n > 0 else 1.0)

        n = int(self.n)
        e = int(self.eliminated_idx)
        if n <= 0:
            return np.ones(1)

        p_judge = np.asarray(self.judge_percent).astype(float)
        s_j = float(np.sum(p_judge))
        p_judge = p_judge / (s_j if s_j > 0 else 1.0)

        rng = np.random.default_rng(0)
        accepted = []

        max_accept = min(1000, 100 * n)
        max_draws = 60000
        tol = 1e-12
        alpha = np.ones(n)

        for _ in range(max_draws):
            p_fan = rng.dirichlet(alpha)
            combined = p_judge + p_fan
            if np.all(combined[e] <= (np.delete(combined, e) + tol)):
                accepted.append(p_fan)
                if len(accepted) >= max_accept:
                    break

        if accepted:
            v = np.mean(np.asarray(accepted), axis=0)
            s = float(np.sum(v))
            return v / (s if s > 0 else 1.0)

        v_base = np.ones(n)
        v_base[e] = 0.3
        return v_base / float(np.sum(v_base))


def build_bayesian_residual_model(industry_idx, age_data):
    with pm.Model() as residual_model:
        industry_effect = pm.Normal(
            "Industry_Effect",
            mu=0,
            sigma=1,
            shape=len(np.unique(industry_idx)),
        )
        age_slope = pm.Normal("Age_Slope", mu=0, sigma=1)
        pref_mu = industry_effect[industry_idx] + age_slope * age_data
        pm.Normal("Delta_V", mu=pref_mu, sigma=0.5, shape=len(age_data))
        return residual_model


class ResidualCalibrationEnsemble:
    def __init__(self, alpha=0.6, eps=1e-9):
        self.alpha = float(alpha)
        self.eps = float(eps)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        exp_x = np.exp(x - float(np.max(x)))
        denom = float(np.sum(exp_x))
        return exp_x / (denom if denom > 0 else 1.0)

    def fuse(self, v_base: np.ndarray, delta_v: np.ndarray) -> np.ndarray:
        v_base = np.asarray(v_base, dtype=float)
        delta_v = np.asarray(delta_v, dtype=float)
        raw = np.log(v_base + self.eps) + self.alpha * delta_v
        return self.softmax(raw)


def compute_week_activity(season_df: pd.DataFrame, week: int) -> pd.DataFrame:
    cols = [c for c in week_judge_cols(week) if c in season_df.columns]
    if not cols:
        return pd.DataFrame()

    base_cols = ["celebrity_name", "results"]
    if "celebrity_age_during_season" in season_df.columns:
        base_cols.append("celebrity_age_during_season")
    if "industry_idx" in season_df.columns:
        base_cols.append("industry_idx")
    tmp = season_df[base_cols + cols].copy()
    tmp["Name"] = tmp["celebrity_name"].map(normalize_name)
    tmp["results_norm"] = tmp["results"].astype(str)

    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    judge_mat = tmp[cols]
    active_judges = judge_mat.notna().sum(axis=1).astype(int)
    judge_points = judge_mat.fillna(0.0).sum(axis=1).astype(float)
    placeholder_zero = (active_judges > 0) & (judge_points == 0.0)

    tmp["active_judges_week"] = active_judges
    tmp["judge_points"] = judge_points
    tmp["placeholder_zero"] = placeholder_zero.astype(int)

    tmp = tmp[~placeholder_zero].copy()

    return tmp.reset_index(drop=True)


def infer_withdrew_exit_week(season_df: pd.DataFrame, max_weeks: int) -> dict[str, int]:
    names = season_df["celebrity_name"].map(normalize_name)
    withdrew_flag = season_df["results"].astype(str).str.contains("Withdrew", case=False, na=False)

    last_active_week = pd.Series(0, index=season_df.index, dtype=int)

    for week in range(1, max_weeks + 1):
        cols = [c for c in week_judge_cols(week) if c in season_df.columns]
        if not cols:
            continue

        mat = season_df[cols].copy()
        for c in cols:
            mat[c] = pd.to_numeric(mat[c], errors="coerce")

        active_judges = mat.notna().sum(axis=1).astype(int)
        judge_points = mat.fillna(0.0).sum(axis=1).astype(float)
        placeholder_zero = (active_judges > 0) & (judge_points == 0.0)
        active = (active_judges > 0) & (~placeholder_zero)

        last_active_week = np.where(active, week, last_active_week)

    exit_week_map: dict[str, int] = {}
    for idx, is_w in withdrew_flag.items():
        if not bool(is_w):
            continue
        n = names.iloc[idx]
        w = int(last_active_week[idx])
        if w > 0:
            exit_week_map[n] = w

    return exit_week_map


def get_elimination_names_for_week(
    week_df: pd.DataFrame,
    week: int,
    withdrew_exit_week: dict[str, int],
) -> list[str]:
    elim = week_df["results_norm"].astype(str).str.contains(
        f"Eliminated Week {week}", case=False, na=False
    )
    withdrew = week_df["Name"].map(lambda n: int(withdrew_exit_week.get(n, -1) == week)).astype(bool)
    elim_names = week_df.loc[elim | withdrew, "Name"].astype(str).tolist()
    return sorted(set(elim_names))


def compare_with_results(results_path: str, new_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    if not os.path.exists(results_path):
        return pd.DataFrame()

    res = pd.read_csv(results_path, encoding="utf-8-sig")
    need_cols = {"Season", "Week", "Name", "Fan_Vote_Percent", "Fan_Vote_Rank"}
    miss = sorted(need_cols - set(res.columns))
    if miss:
        return pd.DataFrame()

    res = res.copy()
    res["Season"] = pd.to_numeric(res["Season"], errors="coerce").astype("Int64")
    res["Week"] = pd.to_numeric(res["Week"], errors="coerce").astype("Int64")
    res["Name"] = res["Name"].map(normalize_name)
    res["Fan_Vote_Percent"] = pd.to_numeric(res["Fan_Vote_Percent"], errors="coerce").fillna(0.0)
    res["Fan_Vote_Rank"] = pd.to_numeric(res["Fan_Vote_Rank"], errors="coerce").fillna(0.0)

    cur = new_df.copy()
    cur["Season"] = pd.to_numeric(cur["Season"], errors="coerce").astype("Int64")
    cur["Week"] = pd.to_numeric(cur["Week"], errors="coerce").astype("Int64")
    cur["Name"] = cur["Name"].map(normalize_name)
    cur["Fan_Vote_Percent"] = pd.to_numeric(cur["Fan_Vote_Percent"], errors="coerce").fillna(0.0)
    cur["Fan_Vote_Rank"] = pd.to_numeric(cur["Fan_Vote_Rank"], errors="coerce").fillna(0.0)

    merged = pd.merge(
        res,
        cur,
        on=["Season", "Week", "Name"],
        how="inner",
        suffixes=("_Results", "_Higher"),
    )
    if merged.empty:
        return merged

    merged["Abs_Error_Percent"] = (merged["Fan_Vote_Percent_Results"] - merged["Fan_Vote_Percent_Higher"]).abs()
    merged["Signed_Error_Percent"] = (
        merged["Fan_Vote_Percent_Higher"] - merged["Fan_Vote_Percent_Results"]
    )
    merged["Abs_Error_Rank"] = (merged["Fan_Vote_Rank_Results"] - merged["Fan_Vote_Rank_Higher"]).abs()
    merged["Source_Script"] = "model1_prediction_higher.py"

    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    return merged


if __name__ == "__main__":
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)
    logging.getLogger("arviz").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Only one chain was sampled.*")

    DATA_PATH = r"d:\MCM_2026_O\data\processed\processed_mcm_wide_clean.csv"

    OUT_PATH = r"d:\MCM_2026_O\data\processed\model1_fan_vote_predictions_higher.csv"
    OUT_WEEK_DIAG_PATH = (
        r"d:\MCM_2026_O\data\processed\model1_fan_vote_predictions_higher_week_diagnostics.csv"
    )
    COMPARE_RESULTS_PATH = r"d:\MCM_2026_O\results\fan_vote_estimates_by_person_week.csv"
    COMPARE_OUT_PATH = (
        r"d:\MCM_2026_O\data\processed\model1_fan_vote_predictions_higher_vs_results.csv"
    )

    MAX_WEEKS = 11
    DRAWS = 100
    TUNE = 100
    CHAINS = 1

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    required_cols = {
        "season",
        "celebrity_name",
        "results",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["celebrity_name"] = df["celebrity_name"].map(normalize_name)
    if "celebrity_age_during_season" not in df.columns:
        df["celebrity_age_during_season"] = 0.0
    if "industry_idx" not in df.columns:
        df["industry_idx"] = 0

    ensemble = ResidualCalibrationEnsemble(alpha=0.6)

    seasons = (
        pd.to_numeric(df["season"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    seasons = sorted(seasons)

    all_preds = []
    diag_rows: list[dict] = []

    for season in seasons:
        season_df = df[df["season"] == season].copy().reset_index(drop=True)
        season_type = get_scoring_system(int(season))
        withdrew_exit_week = infer_withdrew_exit_week(season_df, MAX_WEEKS)

        for week in range(1, MAX_WEEKS + 1):
            week_tmp = compute_week_activity(season_df, week)
            if week_tmp.empty:
                continue

            judge_points = week_tmp["judge_points"].to_numpy(dtype=float)

            if len(week_tmp) < 2:
                diag_rows.append(
                    {
                        "Season": int(season),
                        "Week": int(week),
                        "Season_Type": season_type,
                        "N_active": int(len(week_tmp)),
                        "N_elim": 0,
                        "Elim_constraint_used": 0,
                        "Elim_names": "",
                        "Withdrew_exit_week_count": int(len(withdrew_exit_week)),
                        "Reason": "N_active<2",
                        "Source_Script": "model1_prediction_higher.py",
                    }
                )
                continue

            age = pd.to_numeric(
                week_tmp["celebrity_age_during_season"], errors="coerce"
            ).fillna(0.0).to_numpy()
            age_std = float(np.std(age))
            age_z = (age - float(np.mean(age))) / (age_std if age_std > 0 else 1.0)

            ind_raw = pd.to_numeric(
                week_tmp["industry_idx"], errors="coerce"
            ).fillna(0).astype(int).to_numpy()
            _, ind = np.unique(ind_raw, return_inverse=True)

            elim_names = get_elimination_names_for_week(week_tmp, week, withdrew_exit_week)
            eliminated_idx = None
            if len(elim_names) == 1:
                idxs = np.where(week_tmp["Name"].to_numpy(dtype=str) == elim_names[0])[0]
                if idxs.size > 0:
                    eliminated_idx = int(idxs[0])

            if season_type == "rank":
                judge_ranks = rank_descending(judge_points)
                base_model = RankPhysicalConstraintModel(judge_ranks, eliminated_idx)
                v_base = base_model.get_feasible_fan_rank_prior()
            else:
                s = float(np.sum(judge_points))
                judge_percent = judge_points / (s if s > 0 else 1.0)
                base_model = PercentPhysicalConstraintModel(judge_percent, eliminated_idx)
                v_base = base_model.get_feasible_fan_percent_prior()

            res_model = build_bayesian_residual_model(ind, age_z)
            with res_model:
                trace = pm.sample(
                    DRAWS,
                    tune=TUNE,
                    chains=CHAINS,
                    target_accept=0.9,
                    return_inferencedata=True,
                    compute_convergence_checks=False,
                    progressbar=False,
                )

            delta_v = trace.posterior["Delta_V"].mean(dim=("chain", "draw")).values
            fan_score = ensemble.fuse(v_base, delta_v)

            fan_vote_percent = fan_score / float(
                np.sum(fan_score) if float(np.sum(fan_score)) > 0 else 1.0
            )
            fan_vote_rank = rank_descending(fan_vote_percent)

            names = week_tmp["Name"].to_numpy(dtype=str)

            result_df = pd.DataFrame(
                {
                    "Season": int(season),
                    "Week": int(week),
                    "Name": names,
                    "Fan_Vote_Percent": fan_vote_percent,
                    "Fan_Vote_Rank": fan_vote_rank,
                    "V_Base": v_base,
                    "Delta_V": delta_v,
                    "Judge_Points": judge_points,
                    "Active_Judges_Week": week_tmp["active_judges_week"].to_numpy(dtype=int),
                    "Source_Script": "model1_prediction_higher.py",
                }
            )
            all_preds.append(result_df)

            diag_rows.append(
                {
                    "Season": int(season),
                    "Week": int(week),
                    "Season_Type": season_type,
                    "N_active": int(len(names)),
                    "N_elim": int(len(elim_names)),
                    "Elim_constraint_used": int(eliminated_idx is not None),
                    "Elim_names": "|".join(elim_names),
                    "Withdrew_exit_week_count": int(len(withdrew_exit_week)),
                    "Reason": "",
                    "Source_Script": "model1_prediction_higher.py",
                }
            )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    out_df = (
        pd.concat(all_preds, ignore_index=True)
        if all_preds
        else pd.DataFrame(
            columns=[
                "Season",
                "Week",
                "Name",
                "Fan_Vote_Percent",
                "Fan_Vote_Rank",
                "V_Base",
                "Delta_V",
                "Judge_Points",
                "Active_Judges_Week",
                "Source_Script",
            ]
        )
    )
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT_WEEK_DIAG_PATH, index=False, encoding="utf-8-sig")

    compare_with_results(COMPARE_RESULTS_PATH, out_df, COMPARE_OUT_PATH)

    print(f"Saved predictions: {OUT_PATH} (rows={len(out_df)})")
    print(f"Saved diagnostics: {OUT_WEEK_DIAG_PATH} (rows={len(diag_df)})")
