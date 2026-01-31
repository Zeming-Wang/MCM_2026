import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler

# 1. 加载数据
# 请确保你的文件名与此一致
file_path = '/Users/a202507/Desktop/2026/data/raw/2026_MCM_Problem_C_Data.csv'
df = pd.read_csv(file_path)

def clean_and_transform(df):
    processed_data = df.copy()
    
    # --- 模块一：计算每周平均分 (不丢失任何原始评分) ---
    def get_weekly_mean(row):
        weekly_means = []
        for w in range(1, 12):
            cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
            # 过滤掉 N/A, 0 和 空值
            scores = []
            for col in cols:
                val = row.get(col)
                if pd.notnull(val) and str(val).strip().upper() != 'N/A':
                    try:
                        score = float(val)
                        if score > 0: scores.append(score)
                    except ValueError:
                        continue
            if scores:
                weekly_means.append(np.mean(scores))
        return weekly_means

    processed_data['score_series'] = processed_data.apply(get_weekly_mean, axis=1)

    # --- 模块二：计算学习曲线斜率 (体现导师影响力) ---
    def calculate_slope(series):
        if len(series) < 2: return 0.0 # 样本太少无法计算斜率
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope

    processed_data['growth_slope'] = processed_data['score_series'].apply(calculate_slope)

    # --- 模块三：行业哑变量化 (转化为数值用于因子分析) ---
    # 根据你的 CSV 内容，我们将其归类为四大潜力因子
    processed_data['is_athlete'] = processed_data['celebrity_industry'].str.contains('Athlete|Olympian', case=False, na=False).astype(int)
    processed_data['is_performer'] = processed_data['celebrity_industry'].str.contains('Singer|Rapper|Actor|Dancer', case=False, na=False).astype(int)
    processed_data['is_reality_social'] = processed_data['celebrity_industry'].str.contains('Star|Entrepreneur|Host', case=False, na=False).astype(int)
    
    # --- 模块四：粉丝投票/结果量化 (Success Index) ---
    # 排名越靠前（数值越小），得分越高
    max_p = processed_data['placement'].max()
    processed_data['success_index'] = (max_p - processed_data['placement'] + 1) / max_p

    # --- 模块五：数据归一化准备 ---
    scaler = MinMaxScaler()
    # 对年龄和斜率进行归一化，方便后续因子分析
    processed_data['age_norm'] = scaler.fit_transform(processed_data[['celebrity_age_during_season']])
    
    return processed_data

# 执行清洗
df_final = clean_and_transform(df)

# 保存完整结果（包含所有原始列 + 新增特征列）
output_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Data.csv'
df_final.to_csv(output_path, index=False)

print("数据清洗完成！")
print(f"原始行数: {len(df)}")
print(f"新增特征: growth_slope, is_athlete, is_performer, success_index")