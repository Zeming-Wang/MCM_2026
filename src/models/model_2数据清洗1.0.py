import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
import os

# 1. 加载数据
input_path = '/Users/a202507/Desktop/2026/data/raw/2026_MCM_Problem_C_Data.csv'
output_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Master_v2.csv'
# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

try:
    df = pd.read_csv(input_path)
    print(f"成功加载数据: {input_path}")
except FileNotFoundError:
    print(f"错误：未找到文件 {input_path}")
    exit()

def refined_cleaning(df):
    # --- A. 智能预处理 ---
    # 统一将 N/A 替换为 np.nan 并转换数值
    score_cols = [col for col in df.columns if 'score' in col]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col].replace('N/A', np.nan), errors='coerce')

    # --- B. 身体与竞技特征深度挖掘 ---
    # 1. 身体巅峰偏差指数 (25岁为 1.0, 偏离越多分数越低)
    df['physique_index'] = df['celebrity_age_during_season'].apply(
        lambda x: 1 / (1 + abs(x - 25) * 0.05)
    )

    # 2. 行业背景强度 (根据行业属性赋予不同的“舞蹈迁移潜力”)
    # 运动员加成最高，演艺人员其次，真人秀再次
    industry_weights = {
        'Athlete': 0.9, 'NBA Player': 0.95, 'Olympic': 1.0, 
        'Singer': 0.7, 'Actor': 0.6, 'Actress': 0.6,
        'Reality': 0.3, 'Model': 0.4, 'Entrepreneur': 0.2
    }
    
    def get_industry_power(ind):
        ind_str = str(ind)
        for key, weight in industry_weights.items():
            if key.lower() in ind_str.lower():
                return weight
        return 0.5 # 默认值

    df['dance_potential'] = df['celebrity_industry'].apply(get_industry_power)

    # --- C. 地缘与社交加成 ---
    # 3. 核心地缘因子 (Utah 效应在 DWTS 中极其显著)
    df['geo_bonus'] = df['celebrity_homestate'].apply(lambda x: 1.0 if str(x) == 'Utah' else 0.0)

    # --- D. 学习曲线与技术增长 (核心指标) ---
    def extract_learning_metrics(row):
        weekly_averages = []
        for w in range(1, 12):
            # 动态计算每周 3-4 个评委的均分
            week_cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
            scores = [row[c] for c in week_cols if not np.isnan(row[c]) and row[c] > 0]
            if scores:
                weekly_averages.append(np.mean(scores))
        
        # 稳健斜率计算：至少需要 3 周数据才能形成有效曲线，否则为 0
        if len(weekly_averages) >= 3:
            x = np.arange(len(weekly_averages))
            slope, intercept, r_value, p_value, _ = linregress(x, weekly_averages)
            # 考虑 R-squared (拟合优度)，如果增长不稳定，斜率权重降低
            growth_reliability = slope * (r_value**2) 
        elif len(weekly_averages) == 2:
            growth_reliability = (weekly_averages[1] - weekly_averages[0]) * 0.5
        else:
            growth_reliability = 0.0
            
        return pd.Series([
            growth_reliability, 
            np.mean(weekly_averages) if weekly_averages else 0,
            len(weekly_averages) # 留存周数
        ])

    df[['growth_slope', 'avg_score', 'weeks_lasted']] = df.apply(extract_learning_metrics, axis=1)

    # --- E. 结果归一化 ---
    # 将最终指标缩放到 0-1 之间，方便后续因子分析
    scaler = MinMaxScaler()
    cols_to_scale = ['physique_index', 'dance_potential', 'growth_slope', 'avg_score']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df

# 3. 执行
df_master = refined_cleaning(df)

# 4. 关键特征提取展示
print("\n--- 清洗后核心特征展示 ---")
print(df_master[['celebrity_name', 'growth_slope', 'physique_index', 'geo_bonus']].head())

# 5. 保存
df_master.to_csv(output_path, index=False)
print(f"数据已保存至: {output_path}")