import pandas as pd
import numpy as np
import re

def preprocess_dwts_to_long(raw_csv_path, output_csv_path):

    df = pd.read_csv(raw_csv_path)

    # =========================
    # 1. 提取 judge 列
    # =========================
    judge_cols = [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]

    # =========================
    # 2. wide → long
    # =========================
    df_long = df.melt(
        id_vars=[
            'celebrity_name',
            'ballroom_partner',
            'celebrity_industry',
            'celebrity_age_during_season',
            'season',
            'placement',
            'results'
        ],
        value_vars=judge_cols,
        var_name='week_judge',
        value_name='judge_score'
    )

    # =========================
    # 3. 提取 week / judge
    # =========================
    df_long['week'] = df_long['week_judge'].str.extract(r'week(\d+)_').astype(int)
    df_long['judge_id'] = df_long['week_judge'].str.extract(r'_judge(\d+)_').astype(int)
    df_long.drop(columns='week_judge', inplace=True)

    # =========================
    # 4. 清洗评分
    # =========================
    df_long['judge_score'] = pd.to_numeric(df_long['judge_score'], errors='coerce')

    # 删除真正“无评委”的记录
    df_long = df_long.dropna(subset=['judge_score'])

    # =========================
    # 5. 选手-周粒度的总分
    # =========================
    weekly_total = (
        df_long
        .groupby(['season', 'week', 'celebrity_name'], as_index=False)
        ['judge_score']
        .sum()
        .rename(columns={'judge_score': 'judge_total_score'})
    )

    # =========================
    # 6. 剔除已淘汰选手（在选手粒度）
    # =========================
    weekly_total = weekly_total[weekly_total['judge_total_score'] > 0]

    # =========================
    # 7. 计算 judge_percent（在选手粒度！）
    # =========================
    weekly_total['judge_percent'] = (
        weekly_total
        .groupby(['season', 'week'])['judge_total_score']
        .transform(lambda x: x / x.sum())
    )

    # =========================
    # 8. 合并回 long 表（如主模型仍需要 judge 级信息）
    # =========================
    df_long = df_long.merge(
        weekly_total,
        on=['season', 'week', 'celebrity_name'],
        how='inner'
    )

    # =========================
    # 9. 行业编码
    # =========================
    df_long['celebrity_industry'] = df_long['celebrity_industry'].fillna('Unknown')
    df_long['industry_idx'] = df_long['celebrity_industry'].astype('category').cat.codes

    # =========================
    # 10. 输出
    # =========================
    df_long.to_csv(output_csv_path, index=False)
    print(f"Processed data saved to {output_csv_path}")
    print(f"Rows: {len(df_long)}")

    return df_long
