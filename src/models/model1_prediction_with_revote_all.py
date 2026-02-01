import numpy as np
import pandas as pd
import pymc as pm


def get_scoring_system(season):
    if season <= 2 or season >= 28:
        return "rank"
    return "percent"


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

        max_accept = min(400, 50 * n)
        max_draws = 20000

        for _ in range(max_draws):
            latent = rng.normal(0.0, 1.0, size=n)
            fan_ranks = np.argsort(np.argsort(-latent)) + 1
            rank_sum = judge_ranks + fan_ranks

            if np.all(rank_sum[e] >= np.delete(rank_sum, e)):
                exp_v = np.exp(latent - float(np.max(latent)))
                p = exp_v / float(np.sum(exp_v) if float(np.sum(exp_v)) > 0 else 1.0)
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

        max_accept = min(800, 80 * n)
        max_draws = 40000
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


def build_bayesian_residual_model(industry_idx, age_data, fan_base_data):
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
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def fuse_rank(self, v_base, delta_v):
        raw = v_base + self.alpha * delta_v
        return self.softmax(raw)

    def fuse_percent(self, v_base, delta_v):
        raw = v_base + self.alpha * delta_v
        return self.softmax(raw)

    def adaptive_weight_logic(self, solution_space_entropy):
        self.alpha = 1.0 / (1.0 + np.exp(-solution_space_entropy))
        return self.alpha


def judge_revote_bottom_two(bottom2_indices, judge_scores_bottom2):
    votes = np.zeros(2)
    for scores in judge_scores_bottom2:
        if scores[0] > scores[1]:
            votes[0] += 1
        elif scores[1] > scores[0]:
            votes[1] += 1
    if votes[0] == votes[1]:
        return None
    loser = np.argmin(votes)
    return bottom2_indices[loser]


DEFAULT_SEASONS = [2, 4, 11, 27, 28, 29, 34]
DEFAULT_MAX_WEEKS = 7


def _parse_seasons_env(value: str | None) -> list[int] | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            continue
    return sorted(set(out)) if out else None


if __name__ == "__main__":
    import os
    from pathlib import Path

    project_root = Path(
        os.environ.get("MCM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    ).resolve()

    data_path = str(project_root / "data" / "processed" / "processed_mcm_wide_clean.csv")
    out_path = str(
        project_root
        / "data"
        / "processed"
        / "model1_fan_vote_predictions_with_revote_all.csv"
    )

    max_weeks = int(os.environ.get("MCM_MAX_WEEKS", str(DEFAULT_MAX_WEEKS)))
    draws = int(os.environ.get("MCM_DRAWS", "200"))
    tune = int(os.environ.get("MCM_TUNE", "200"))
    chains = int(os.environ.get("MCM_CHAINS", "1"))

    def _week_judge_cols(week):
        return [
            f"week{week}_judge1_score",
            f"week{week}_judge2_score",
            f"week{week}_judge3_score",
            f"week{week}_judge4_score",
        ]

    def _rank_descending(values):
        values = np.asarray(values)
        return np.argsort(np.argsort(-values)) + 1

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    required_cols = {
        "season",
        "celebrity_name",
        "celebrity_age_during_season",
        "industry_idx",
        "results",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    ensemble = ResidualCalibrationEnsemble()

    seasons = (
        pd.to_numeric(df["season"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    seasons = sorted(seasons)
    seasons_filter = _parse_seasons_env(os.environ.get("MCM_SEASONS"))
    if seasons_filter is None:
        seasons_filter = DEFAULT_SEASONS
    seasons = [s for s in seasons if int(s) in set(seasons_filter)]

    all_preds = []

    for season in seasons:
        season_df = df[df["season"] == season].copy().reset_index(drop=True)
        season_type = get_scoring_system(season)

        print(f"Season={season}, Type={season_type}")

        for week in range(1, max_weeks + 1):
            judge_cols = [c for c in _week_judge_cols(week) if c in season_df.columns]
            if not judge_cols:
                continue

            week_df = season_df[
                [
                    "celebrity_name",
                    "celebrity_age_during_season",
                    "industry_idx",
                    "results",
                ]
                + judge_cols
            ].copy()

            for col in judge_cols:
                week_df[col] = pd.to_numeric(week_df[col], errors="coerce").fillna(0.0)

            judge_points = week_df[judge_cols].sum(axis=1).to_numpy()
            active_mask = judge_points > 0
            week_df = week_df[active_mask].reset_index(drop=True)
            judge_points = judge_points[active_mask]

            if len(week_df) == 0:
                continue
            if len(week_df) == 1:
                break

            age = pd.to_numeric(
                week_df["celebrity_age_during_season"], errors="coerce"
            ).fillna(0.0).to_numpy()
            age_std = float(np.std(age))
            age_z = (age - float(np.mean(age))) / (age_std if age_std > 0 else 1.0)

            ind_raw = pd.to_numeric(
                week_df["industry_idx"], errors="coerce"
            ).fillna(0).astype(int).to_numpy()
            _, ind = np.unique(ind_raw, return_inverse=True)

            eliminated_flag = (
                week_df["results"]
                .astype(str)
                .str.contains(f"Eliminated Week {week}", case=False, na=False)
                .astype(int)
                .to_numpy()
            )
            actual_elim = week_df.loc[
                eliminated_flag.astype(bool), "celebrity_name"
            ].tolist()

            eliminated_idx = None
            if int(np.sum(eliminated_flag)) > 0:
                eliminated_idx = int(np.where(eliminated_flag == 1)[0][0])

            if season_type == "rank":
                judge_ranks = _rank_descending(judge_points)
                base_model = RankPhysicalConstraintModel(judge_ranks, eliminated_idx)
                v_base = base_model.get_feasible_fan_rank_prior()
            else:
                s = float(np.sum(judge_points))
                judge_percent = judge_points / (s if s > 0 else 1.0)
                base_model = PercentPhysicalConstraintModel(judge_percent, eliminated_idx)
                v_base = base_model.get_feasible_fan_percent_prior()

            res_model = build_bayesian_residual_model(ind, age_z, v_base)
            with res_model:
                trace = pm.sample(
                    draws,
                    tune=tune,
                    chains=chains,
                    target_accept=0.9,
                    return_inferencedata=True,
                    progressbar=False,
                )

            delta_v = trace.posterior["Delta_V"].mean(dim=("chain", "draw")).values

            if season_type == "rank":
                fan_score = ensemble.fuse_rank(v_base, delta_v)
            else:
                fan_score = ensemble.fuse_percent(v_base, delta_v)

            names = week_df["celebrity_name"].to_numpy()

            fan_vote_percent = fan_score / float(
                np.sum(fan_score) if float(np.sum(fan_score)) > 0 else 1.0
            )
            fan_vote_rank = _rank_descending(fan_vote_percent)
            judge_rank = _rank_descending(judge_points)
            judge_percent = judge_points / float(
                np.sum(judge_points) if float(np.sum(judge_points)) > 0 else 1.0
            )

            score_cols = {}
            for idx in range(4):
                col = f"week{week}_judge{idx + 1}_score"
                key = f"Judge{idx + 1}_Score"
                if col in week_df.columns:
                    score_cols[key] = week_df[col].to_numpy()
                else:
                    score_cols[key] = np.zeros(len(week_df))

            if season_type == "rank":
                bottom2_indices = np.argsort(fan_vote_rank)[-2:].tolist()
            else:
                bottom2_indices = np.argsort(fan_vote_percent)[:2].tolist()

            judge_scores_bottom2 = []
            for col in judge_cols:
                judge_scores_bottom2.append(week_df.loc[bottom2_indices, col].to_numpy())
            judge_scores_bottom2 = np.asarray(judge_scores_bottom2)

            eliminated_pred_idx = int(np.argmin(fan_vote_percent))
            revote_elim = judge_revote_bottom_two(bottom2_indices, judge_scores_bottom2)
            if revote_elim is not None:
                eliminated_pred_idx = int(revote_elim)

            predicted_elim_mask = np.zeros(len(week_df), dtype=int)
            predicted_elim_mask[eliminated_pred_idx] = 1

            result_df = pd.DataFrame(
                {
                    "Season": season,
                    "Week": week,
                    "Season_Type": season_type,
                    "Name": names,
                    "Judge_Points": judge_points,
                    "Judge_Rank": judge_rank,
                    "Judge_Percent": judge_percent,
                    **score_cols,
                    "Fan_Vote_Percent": fan_vote_percent,
                    "Fan_Vote_Rank": fan_vote_rank,
                    "Predicted_Eliminated": predicted_elim_mask,
                    "Actual_Eliminated": eliminated_flag,
                    "V_Base": v_base,
                    "Delta_V": delta_v,
                }
            )
            all_preds.append(result_df)

            if len(actual_elim) > 0:
                predicted_elim_name = str(names[eliminated_pred_idx])
                print(
                    f"\nWeek={week}, Predicted_Eliminated={predicted_elim_name}, Actual_Eliminated={actual_elim}"
                )
                print(result_df.sort_values("Fan_Vote_Percent", ascending=True).head(5))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if all_preds:
        out_df = pd.concat(all_preds, ignore_index=True)
    else:
        out_df = pd.DataFrame(
            columns=[
                "Season",
                "Week",
                "Name",
                "Fan_Vote_Percent",
                "Fan_Vote_Rank",
                "V_Base",
                "Delta_V",
            ]
        )

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions: {out_path} (rows={len(out_df)})")
