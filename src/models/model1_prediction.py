import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import networkx as nx
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
import importlib.util
from pathlib import Path

# ==========================================
# 1. 基础物理约束层 (Base Layer - Model A)
# ==========================================
class PhysicalConstraintModel:
    def __init__(self, judges_scores, season):
        self.judges_scores = judges_scores
        self.season = season

    """
    修改说明：
    不再返回全 1 的平均分布，而是直接计算当周选手的法官评分百分比。
    这是遵循题目 PDF 中 Jennie Garth 案例（29/117 = 24.8%）的核心逻辑。
    """
    def get_weekly_judge_percent(self):
        total_sum = np.sum(self.judges_scores)
        if total_sum == 0:
            return np.ones(len(self.judges_scores)) / len(self.judges_scores)
        return self.judges_scores / total_sum


def _get_scoring_mode(season):
    if season <= 2 or season >= 28:
        return "rank"
    return "percent"


def _week_judge_cols(week):
    return [
        f"week{week}_judge1_score",
        f"week{week}_judge2_score",
        f"week{week}_judge3_score",
        f"week{week}_judge4_score",
    ]
"""注意这里取了四个评委的分数，可能要结合数据清洗文件进行修改"""

def _rank_descending(values):
    values = np.asarray(values)
    # 1 = 最好（数值越大排名越靠前）
    return np.argsort(np.argsort(-values)) + 1
# ==========================================
# 2. 贝叶斯残差扰动层 (Residual Layer - Model B)
# ==========================================
def build_bayesian_residual_model(industry_idx, age_data, fan_base_data):
    with pm.Model() as residual_model:
        industry_effect = pm.Normal('Industry_Effect', mu=0, sigma=1, shape=len(np.unique(industry_idx)))
        age_slope = pm.Normal('Age_Slope', mu=0, sigma=1)
        
        pref_mu = industry_effect[industry_idx] + age_slope * age_data
        audience_pref = pm.Normal('Audience_Preference', mu=pt.mean(pref_mu), sigma=1)
        
        """
        修改说明：
        Delta_V 现在代表“粉丝投票意向”。
        在 fuse 阶段，它会被转化为 Fan Percent (P_f)。
        """
        delta_v = pm.Normal('Delta_V', mu=audience_pref, sigma=0.5, shape=len(age_data))
        return residual_model

# ==========================================
# 3. 异构融合架构 (Ensemble Engine)
# ==========================================
class ResidualCalibrationEnsemble:
    def __init__(self, alpha_init=0.5):
        self.alpha = alpha_init 

  
    def fuse_by_percentage(self, p_judge, delta_v_samples):
        # 1. 计算残差均值并映射为粉丝百分比 P_f (和为1)
        delta_v_mean = np.mean(delta_v_samples, axis=0)
        exp_v = np.exp(delta_v_mean * self.alpha)
        p_fan = exp_v / np.sum(exp_v)
        
        # 2. 物理加和逻辑：Sum of Percents
        v_final = p_judge + p_fan
        return v_final, p_fan

    """
    修改说明：
    针对 Season 1-2 的排名制逻辑。
    Total = Rank_Judge + Rank_Fan，取加和最高者（或秩和最小者）存活。
    """
    def fuse_by_ranking(self, p_judge, delta_v_samples):
        # 将法官百分比转为排名 (分数越高排名越小, e.g., 1st, 2nd)
        rank_j = len(p_judge) - np.argsort(np.argsort(p_judge))
        
        delta_v_mean = np.mean(delta_v_samples, axis=0)
        rank_f = len(delta_v_mean) - np.argsort(np.argsort(delta_v_mean))
        
        return (rank_j + rank_f), rank_f

    def adaptive_weight_logic(self, solution_space_entropy):
        self.alpha = 1.0 / (1.0 + np.exp(-solution_space_entropy))
        return self.alpha


def _fan_percent_from_delta(delta_v_samples, alpha):
    delta_v_mean = np.mean(delta_v_samples, axis=0)
    exp_v = np.exp(delta_v_mean * alpha)
    total = np.sum(exp_v)
    if total == 0:
        return np.ones_like(exp_v) / len(exp_v)
    return exp_v / total

    """目前只能将模型产生的一个粉丝映射进行固定，避免散落"""

# ==========================================
# 5. 主逻辑执行流程
# ==========================================
if __name__ == "__main__":
    import os
    DATA_PATH = r"d:\MCM_2026_O\data\processed\processed_mcm_data.csv"
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        
        CURRENT_SEASON = 1
        MAX_WEEKS = 11
        DRAWS = 40
        TUNE = 40
        CHAINS = 1
        ENTROPY_VALUE = 0.61

        season_df = df[df["season"] == CURRENT_SEASON].copy().reset_index(drop=True)
        mode = _get_scoring_mode(CURRENT_SEASON)

        print(f"Season={CURRENT_SEASON}, Mode={mode}")

        for week in range(1, MAX_WEEKS + 1):
            judge_cols = [c for c in _week_judge_cols(week) if c in season_df.columns]
            if not judge_cols:
                continue

            week_df = season_df[
                ["celebrity_name", "celebrity_age_during_season", "industry_idx", "results"]
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

            p_judge = PhysicalConstraintModel(judge_points, CURRENT_SEASON).get_weekly_judge_percent()

            age = pd.to_numeric(
                week_df["celebrity_age_during_season"], errors="coerce"
            ).fillna(0.0).to_numpy()
            age_std = float(np.std(age))
            age_z = (age - float(np.mean(age))) / (age_std if age_std > 0 else 1.0)

            ind_raw = pd.to_numeric(
                week_df["industry_idx"], errors="coerce"
            ).fillna(0).astype(int).to_numpy()
            _, ind = np.unique(ind_raw, return_inverse=True)

            res_model = build_bayesian_residual_model(ind, age_z, np.zeros_like(age_z))
            with res_model:
                trace = pm.sample(
                    DRAWS,
                    tune=TUNE,
                    chains=CHAINS,
                    return_inferencedata=True,
                    progressbar=False,
                )

            delta_v_samples = (
                trace.posterior["Delta_V"].values.reshape(-1, len(week_df))
            )

            ensemble = ResidualCalibrationEnsemble()
            alpha = ensemble.adaptive_weight_logic(ENTROPY_VALUE)
            p_fan = _fan_percent_from_delta(delta_v_samples, alpha)

            names = week_df["celebrity_name"].to_numpy()
            actual_elim_mask = (
                week_df["results"]
                .astype(str)
                .str.contains(f"Eliminated Week {week}", case=False, na=False)
                .to_numpy()
            )
            actual_elim = week_df.loc[actual_elim_mask, "celebrity_name"].tolist()

            if mode == "percent":
                total = p_judge + p_fan
                eliminated_idx = int(np.argmin(total))
                metric_name = "Percent_Sum"
                final_metric = total
                ascending_flag = True
            else:
                rank_j = _rank_descending(judge_points)
                rank_f = _rank_descending(p_fan)
                rank_sum = rank_j + rank_f

                """注意，这里的评委分在csv文件中没有呈现，只能暂时使用已有的评委分数"""
                if CURRENT_SEASON >= 28 and len(rank_sum) >= 2:
                    bottom2 = np.argsort(rank_sum)[-2:]
                    eliminated_idx = int(bottom2[np.argmin(judge_points[bottom2])])
                    metric_name = "Rank_Sum_Revote"
                else:
                    eliminated_idx = int(np.argmax(rank_sum))
                    metric_name = "Rank_Sum"

                final_metric = rank_sum
                ascending_flag = True

            result_df = pd.DataFrame(
                {
                    "Week": week,
                    "Name": names,
                    "Judge_P": p_judge,
                    "Fan_P": p_fan,
                    metric_name: final_metric,
                }
            )

            predicted_elim_name = str(names[eliminated_idx])
            print(f"\nWeek={week}, Predicted_Eliminated={predicted_elim_name}, Actual_Eliminated={actual_elim}")
            print(result_df.sort_values(metric_name, ascending=ascending_flag).head(5))

        """
        后续分析：此处可调用之前的可视化模块进行 Check
        """
    else:
        print(f"Error: Data file not found.")
