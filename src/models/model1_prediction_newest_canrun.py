import numpy as np
import pandas as pd
import pymc as pm

# ==========================================
# 0. 赛制判断函数
# ==========================================
def get_scoring_system(season):
    """
    🆕 NEW
    根据赛季返回赛制类型
    """
    if season <= 2 or season >= 28:
        return "rank"
    return "percent"

# ==========================================
# 1. Base Layer：可行解空间约束
# ==========================================

class RankPhysicalConstraintModel:
    """
    ⚠️ MODIFIED
    排名赛制下的 Base Layer
    利用已知淘汰者，缩小粉丝排名的可行解空间
    """
    def __init__(self, judge_ranks, eliminated_idx=None):
        self.judge_ranks = np.asarray(judge_ranks)
        self.eliminated_idx = eliminated_idx
        self.n = len(judge_ranks)

    def get_feasible_fan_rank_prior(self):
        """
        返回一个满足约束条件的 Fan Rank 期望分布
        """
        if self.eliminated_idx is None:
            return np.ones(self.n) / (self.n if self.n > 0 else 1.0)

        feasible_scores = np.zeros(self.n)
        for i in range(self.n):
            if i == self.eliminated_idx:
                feasible_scores[i] = 1.0
            else:
                feasible_scores[i] = 0.5
        exp_v = np.exp(-feasible_scores)
        return exp_v / exp_v.sum()


class PercentPhysicalConstraintModel:
    """
    ⚠️ MODIFIED
    百分赛制下的 Base Layer
    """
    def __init__(self, judge_percent, eliminated_idx=None):
        self.judge_percent = np.asarray(judge_percent)
        self.eliminated_idx = eliminated_idx
        self.n = len(judge_percent)

    def get_feasible_fan_percent_prior(self):
        """
        返回一个满足不等式约束的粉丝百分比先验
        """
        v_base = np.ones(self.n)
        if self.eliminated_idx is not None:
            v_base[self.eliminated_idx] = 0.3
        return v_base / v_base.sum()


# ==========================================
# 2. 贝叶斯残差扰动层 (Residual Layer - Model B)
# ==========================================
def build_bayesian_residual_model(industry_idx, age_data, fan_base_data):
    """
    构建贝叶斯网络：Industry/Age -> Preference/Volatility -> Score_Residual
    """
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


# ==========================================
# 3. 融合层
# ==========================================
class ResidualCalibrationEnsemble:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def fuse_rank(self, v_base, delta_v):
        """
        ⚠️ MODIFIED
        排名赛制融合：v_base + αΔV → fan ranking score
        """
        raw = v_base + self.alpha * delta_v
        return self.softmax(raw)

    def fuse_percent(self, v_base, delta_v):
        """
        ⚠️ MODIFIED
        百分赛制融合
        """
        raw = v_base + self.alpha * delta_v
        return self.softmax(raw)

    def adaptive_weight_logic(self, solution_space_entropy):
        """
        根据解空间熵动态调整 alpha
        """
        self.alpha = 1.0 / (1.0 + np.exp(-solution_space_entropy))
        return self.alpha


# ==========================================
# 4. Season ≥28：Bottom-2 Judge Revote
# ==========================================
def judge_revote_bottom_two(bottom2_indices, judge_scores_bottom2):
    """
    🆕 NEW
    Season ≥ 28 的评委复活投票机制
    """
    votes = np.zeros(2)
    for scores in judge_scores_bottom2:
        if scores[0] > scores[1]:
            votes[0] += 1
        elif scores[1] > scores[0]:
            votes[1] += 1
    if votes[0] == votes[1]:
        return None  # 平票，不淘汰
    loser = np.argmin(votes)
    return bottom2_indices[loser]


# ==========================================
# 5. 主逻辑示例
# ==========================================
if __name__ == "__main__":
    import os

    DATA_PATH = r"d:\MCM_2026_O\data\processed\processed_mcm_wide_clean.csv"

    OUT_PATH = r"d:\MCM_2026_O\data\processed\model1_fan_vote_predictions.csv"

    MAX_WEEKS = 11
    DRAWS = 200
    TUNE = 200
    CHAINS = 1

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

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
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

    all_preds = []

    for season in seasons:
        season_df = df[df["season"] == season].copy().reset_index(drop=True)
        season_type = get_scoring_system(season)

        print(f"Season={season}, Type={season_type}")

        for week in range(1, MAX_WEEKS + 1):
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

        if len(week_df) <= 1:
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
        actual_elim = week_df.loc[eliminated_flag.astype(bool), "celebrity_name"].tolist()

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
                DRAWS,
                tune=TUNE,
                chains=CHAINS,
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

        fan_vote_percent = fan_score / float(np.sum(fan_score) if float(np.sum(fan_score)) > 0 else 1.0)
        fan_vote_rank = _rank_descending(fan_vote_percent)

        result_df = pd.DataFrame(
            {
                "Season": season,
                "Week": week,
                "Name": names,
                "Fan_Vote_Percent": fan_vote_percent,
                "Fan_Vote_Rank": fan_vote_rank,
                "V_Base": v_base,
                "Delta_V": delta_v,
            }
        )
        all_preds.append(result_df)

        if len(actual_elim) > 0:
            eliminated_pred_idx = int(np.argmin(fan_vote_percent))
            metric_name = "Fan_Vote_Percent"

            if season_type == "rank" and season >= 28 and len(names) >= 2:
                bottom2_indices = np.argsort(fan_vote_percent)[:2].tolist()
                judge_scores_bottom2 = []
                for col in judge_cols:
                    judge_scores_bottom2.append(
                        week_df.loc[bottom2_indices, col].to_numpy()
                    )
                judge_scores_bottom2 = np.asarray(judge_scores_bottom2)

                revote_elim = judge_revote_bottom_two(
                    bottom2_indices, judge_scores_bottom2
                )
                if revote_elim is not None:
                    eliminated_pred_idx = int(revote_elim)
                    metric_name = "Fan_Vote_Percent_Bottom2_Revote"

            predicted_elim_name = str(names[eliminated_pred_idx])
            print(
                f"\nWeek={week}, Predicted_Eliminated={predicted_elim_name}, Actual_Eliminated={actual_elim}"
            )
            print(result_df.sort_values("Fan_Vote_Percent", ascending=True).head(5))

    if all_preds:
        out_df = pd.concat(all_preds, ignore_index=True)
        out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
        print(f"Saved predictions: {OUT_PATH} (rows={len(out_df)})")
    else:
        print("No predictions generated.")
