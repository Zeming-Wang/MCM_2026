import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import rankdata
import os

# ==========================================
# 0. 配置
# ==========================================
DATA_PATH = r"d:\MCM_2026_O\data\processed\processed_dwts_long.csv"
TARGET_SEASON = 1   # 可切换 season
MAX_WEEKS = 11      # 不同赛季自动适配

# ==========================================
# 1. Model B：粉丝投票反演模型
# ==========================================
def build_fan_vote_model(industry_idx, age, eliminated_flag):
    """
    industry_idx : (N,)
    age          : (N,)
    eliminated   : (N,)  1 = 本周被淘汰, 0 = 存活
    """

    age_z = (age - age.mean()) / age.std()

    with pm.Model() as model:
        # 行业效应
        industry_effect = pm.Normal(
            "industry_effect",
            mu=0,
            sigma=1,
            shape=len(np.unique(industry_idx))
        )

        # 年龄效应
        age_slope = pm.Normal("age_slope", 0, 1)

        # 粉丝潜变量（未归一化）
        fan_latent = industry_effect[industry_idx] + age_slope * age_z

        # Fan percent（归一化）
        fan_percent = pm.Deterministic(
            "fan_percent",
            pm.math.softmax(fan_latent)
        )

        # Likelihood：淘汰更可能发生在 fan_latent 低的人
        pm.Bernoulli(
            "elimination_obs",
            logit_p=-fan_latent,
            observed=eliminated_flag
        )

    return model


# ==========================================
# 2. 融合规则（严格对齐官方）
# ==========================================
def fuse_scores(judge_percent, fan_percent, season):
    """
    返回 combined_score, metric_type
    """
    if season <= 2:
        # Rank-based
        rank_j = rankdata(-judge_percent, method="average")
        rank_f = rankdata(-fan_percent, method="average")
        combined = rank_j + rank_f
        return combined, "rank"
    else:
        # Percent-based
        combined = judge_percent + fan_percent
        return combined, "percent"


# ==========================================
# 3. 主执行流程（逐周）
# ==========================================
def run_season_model(df, season):

    season_df = df[df["season"] == season].copy()
    weeks = sorted(season_df["week"].unique())

    print(f"\nRunning Season {season}, Weeks: {weeks}")

    all_results = []

    for week in weeks:

        week_df = season_df[season_df["week"] == week].copy()

        # 若只有 1 人，比赛结束
        if week_df["celebrity_name"].nunique() <= 1:
            break

        # ==========================
        # 数据提取
        # ==========================
        judge_percent = (
            week_df
            .drop_duplicates("celebrity_name")
            .sort_values("celebrity_name")["judge_percent"]
            .values
        )

        age = (
            week_df
            .drop_duplicates("celebrity_name")
            .sort_values("celebrity_name")["celebrity_age_during_season"]
            .values
        )

        industry_idx = (
            week_df
            .drop_duplicates("celebrity_name")
            .sort_values("celebrity_name")["industry_idx"]
            .values
        )

        # 本周是否被淘汰（placement / results 可推）
        eliminated_flag = (
            week_df
            .drop_duplicates("celebrity_name")
            .sort_values("celebrity_name")["results"]
            .str.contains("Eliminated", case=False)
            .astype(int)
            .values
        )

        names = (
            week_df
            .drop_duplicates("celebrity_name")
            .sort_values("celebrity_name")["celebrity_name"]
            .values
        )

        # ==========================
        # 贝叶斯反演粉丝投票
        # ==========================
        model = build_fan_vote_model(industry_idx, age, eliminated_flag)
        with model:
            trace = pm.sample(
                500,
                tune=500,
                chains=2,
                target_accept=0.9,
                progressbar=False
            )

        fan_percent = trace.posterior["fan_percent"].mean(
            dim=("chain", "draw")
        ).values

        # ==========================
        # 融合
        # ==========================
        combined, mode = fuse_scores(judge_percent, fan_percent, season)

        # ==========================
        # 输出
        # ==========================
        result = pd.DataFrame(
            {
                "Season": season,
                "Week": week,
                "Name": names,
                "Judge_Percent": judge_percent,
                "Fan_Percent": fan_percent,
                "Combined": combined,
            }
        )

        ascending = True if mode == "rank" else False
        result = result.sort_values("Combined", ascending=ascending)

        print(f"\nSeason {season} - Week {week} ({mode})")
        print(result.head(5))

        all_results.append(result)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_dir = r"d:\MCM_2026_O\data\processed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"season{season}_fan_fusion_results.csv"
        )
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved season {season} results to {output_path}")


# ==========================================
# 4. 启动
# ==========================================
if __name__ == "__main__":
    df_long = pd.read_csv(DATA_PATH)
    run_season_model(df_long, TARGET_SEASON)