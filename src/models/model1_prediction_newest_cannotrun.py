import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

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
    def __init__(self, judge_ranks, eliminated_idx):
        self.judge_ranks = np.asarray(judge_ranks)
        self.eliminated_idx = eliminated_idx
        self.n = len(judge_ranks)

    def get_feasible_fan_rank_prior(self):
        """
        返回一个满足约束条件的 Fan Rank 期望分布
        """
        feasible_scores = np.zeros(self.n)
        for i in range(self.n):
            if i == self.eliminated_idx:
                feasible_scores[i] = 1.0  # 淘汰者粉丝排名最差
            else:
                feasible_scores[i] = 0.5  # 其余选手优于淘汰者
        exp_v = np.exp(-feasible_scores)
        return exp_v / exp_v.sum()


class PercentPhysicalConstraintModel:
    """
    ⚠️ MODIFIED
    百分赛制下的 Base Layer
    """
    def __init__(self, judge_percent, eliminated_idx):
        self.judge_percent = np.asarray(judge_percent)
        self.eliminated_idx = eliminated_idx
        self.n = len(judge_percent)

    def get_feasible_fan_percent_prior(self):
        """
        返回一个满足不等式约束的粉丝百分比先验
        """
        v_base = np.ones(self.n)
        v_base[self.eliminated_idx] = 0.3  # 淘汰者粉丝占比应更低
        return v_base / v_base.sum()


# ==========================================
# 2. 贝叶斯残差扰动层 (Residual Layer - Model B)
# ==========================================
def build_bayesian_residual_model(industry_idx, age_data, fan_base_data):
    """
    构建贝叶斯网络：Industry/Age -> Preference/Volatility -> Score_Residual
    """
    with pm.Model() as residual_model:
        industry_effect = pm.Normal('Industry_Effect', mu=0, sigma=1, shape=len(np.unique(industry_idx)))
        age_slope = pm.Normal('Age_Slope', mu=0, sigma=1)
        pref_mu = industry_effect[industry_idx] + age_slope * age_data
        audience_pref = pm.Normal('Audience_Preference', mu=pt.mean(pref_mu), sigma=1)
        delta_v = pm.Normal('Delta_V', mu=audience_pref, sigma=0.5, shape=len(age_data))
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
    # 模拟数据
    season = 29
    judge_ranks = [1, 2, 3, 4]  # 排名制示例
    judge_percent = [0.25, 0.25, 0.25, 0.25]  # 百分制示例
    eliminated_idx = 3  # 已知淘汰者索引

    age_mock = np.array([24, 35, 29, 41])
    ind_mock = np.array([0, 1, 0, 2])
    fb_mock = np.array([0.1, 0.4, 0.3, 0.2])

    # 判定赛制
    season_type = get_scoring_system(season)
    ensemble = ResidualCalibrationEnsemble()

    if season_type == "rank":
        base_model = RankPhysicalConstraintModel(judge_ranks, eliminated_idx)
        v_base = base_model.get_feasible_fan_rank_prior()
        # 构建贝叶斯模型
        res_model = build_bayesian_residual_model(ind_mock, age_mock, fb_mock)
        with res_model:
            trace = pm.sample(100, tune=50, chains=1, progressbar=False)
        delta_v_samples = trace.posterior['Delta_V'].values[0]
        fan_score = ensemble.fuse_rank(v_base, delta_v_samples)

        # 🆕 Revote 机制 (Season >=28)
        if season >= 28:
            # 假设取最后两名索引
            bottom2_indices = [2, 3]
            judge_scores_bottom2 = np.array([[8, 7], [6, 6], [5, 4]])  # 示例: 3位评委打分
            eliminated = judge_revote_bottom_two(bottom2_indices, judge_scores_bottom2)
            print("Judge Revote Eliminated:", eliminated)

    else:
        base_model = PercentPhysicalConstraintModel(judge_percent, eliminated_idx)
        v_base = base_model.get_feasible_fan_percent_prior()
        res_model = build_bayesian_residual_model(ind_mock, age_mock, fb_mock)
        with res_model:
            trace = pm.sample(100, tune=50, chains=1, progressbar=False)
        delta_v_samples = trace.posterior['Delta_V'].values[0]
        fan_score = ensemble.fuse_percent(v_base, delta_v_samples)

    print("Fan Score / Fan Ranking Probabilities:", fan_score)
