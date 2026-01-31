import pandas as pd
import numpy as np
import os

def optimized_cleaning_for_mcm(df):
    # 1. 统一处理缺失值：区分“不复存在”与“数据丢失”
    judge_cols = [
        'week1_judge1_score',
        'week1_judge2_score',
        'week1_judge3_score',
        'week1_judge4_score',
    ]

    all_judge_cols = []
    for week in range(1, 12):
        for j in range(1, 5):
            col = f"week{week}_judge{j}_score"
            if col in df.columns:
                all_judge_cols.append(col)

    for col in all_judge_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['active_judges'] = df[judge_cols].notnull().sum(axis=1)
    
    # 2. 核心：计算“法官百分比 (Judges Score Percent)”
    # 无论哪个赛季，统一转化为百分比，这是 Model A 物理约束的基础
    def calculate_judge_percent(group):
        # 填充 NaN 为 0 用于求和，或者根据具体业务逻辑处理
        total_scores = group[judge_cols].fillna(0).sum(axis=1)
        group_total = total_scores.sum()
        if group_total > 0:
            group['judge_percent'] = total_scores / group_total
        else:
            group['judge_percent'] = 0
        return group

    # 必须按“赛季+周”分组计算，因为淘汰是每周发生的
    # 注意：原始数据结构是每行一个选手的一个赛季结果，包含了所有周的分数。
    # 这里原来的逻辑可能假设了长格式数据，但根据 CSV 表头，是宽格式（week1_... week2_...）。
    # 如果要对每周进行处理，需要 melt 或者针对特定周处理。
    # 为了简化，我们暂时只处理 Week 1 作为示例，或者将数据转换为长格式。
    
    # 简单起见，我们这里只计算 Week 1 的百分比，实际应用可能需要 reshape
    df = df.groupby(['season']).apply(calculate_judge_percent)

    # 3. 特征工程优化：为 Model B (残差层) 准备输入
    # 粉丝狂热度预估：职业(industry) + 年龄(age) 的交互项
    df['youth_factor'] = (df['celebrity_age_during_season'] < 30).astype(int)
    # 行业影响力（Embedding 的前置清洗）
    df['is_musician'] = df['celebrity_industry'].str.contains('Singer|Musician', case=False, na=False).astype(int)
    
    # 添加 industry_idx 供贝叶斯模型使用
    df['celebrity_industry'] = df['celebrity_industry'].fillna('Unknown')
    df['industry_idx'] = df['celebrity_industry'].astype('category').cat.codes

    # 4. 数据平滑：处理多舞种周
    # 有些周选手跳两支舞，数据会波动，需取平均或合并
    return df


def build_weekly_long(df, max_weeks=11):
    keep_cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "season",
        "results",
        "placement",
        "youth_factor",
        "is_musician",
        "industry_idx",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    frames = []
    for week in range(1, max_weeks + 1):
        week_cols = [
            f"week{week}_judge1_score",
            f"week{week}_judge2_score",
            f"week{week}_judge3_score",
            f"week{week}_judge4_score",
        ]
        week_cols = [c for c in week_cols if c in df.columns]
        if not week_cols:
            continue

        tmp = df[keep_cols + week_cols].copy()
        for col in week_cols:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        tmp["week"] = week
        tmp["active_judges_week"] = tmp[week_cols].notnull().sum(axis=1)
        tmp["judge_points"] = tmp[week_cols].fillna(0.0).sum(axis=1)
        tmp["is_active"] = (tmp["judge_points"] > 0).astype(int)
        tmp["eliminated_flag"] = (
            tmp["results"]
            .astype(str)
            .str.contains(f"Eliminated Week {week}", case=False, na=False)
            .astype(int)
        )

        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)
    long_df["judge_percent"] = (
        long_df.groupby(["season", "week"])["judge_points"]
        .transform(lambda s: s / s.sum() if float(s.sum()) > 0 else 0.0)
        .fillna(0.0)
    )

    return long_df

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "2026_MCM_Problem_C_Data.csv")
    processed_dir = os.path.join("data", "processed")
    processed_path = os.path.join(processed_dir, "processed_mcm_data.csv")

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    if os.path.exists(raw_path):
        print(f"Reading data from {raw_path}...")
        df = pd.read_csv(raw_path)
        processed_df = optimized_cleaning_for_mcm(df)
        processed_df.to_csv(processed_path, index=False)
        print(f"Data processed and saved to {processed_path}")
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Columns: {processed_df.columns.tolist()}")

        weekly_long_path = os.path.join(processed_dir, "processed_mcm_weekly_long.csv")
        weekly_long_df = build_weekly_long(processed_df, max_weeks=11)
        weekly_long_df.to_csv(weekly_long_path, index=False)
        print(f"Weekly long data saved to {weekly_long_path}")
        print(f"Weekly long data shape: {weekly_long_df.shape}")
    else:
        print(f"Error: Raw data file not found at {raw_path}")

#