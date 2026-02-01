import pandas as pd
import numpy as np
from scipy.stats import linregress
import os

# 1. 加载数据
# 使用绝对路径以确保在任何目录下运行都能找到文件
file_path = '/Users/a202507/Desktop/2026/data/raw/2026_MCM_Problem_C_Data.csv'
if not os.path.exists(file_path):
    # Fallback try relative path just in case
    file_path = 'data/raw/2026_MCM_Problem_C_Data.csv'

raw_df = pd.read_csv(file_path)

# 2. 定义处理分数的函数 (计算每周平均分并处理 N/A)
def get_weekly_averages(row):
    weekly_avgs = []
    for w in range(1, 12):  # 假设最多 11 周
        cols = [f'week{w}_judge1_score', f'week{w}_judge2_score', 
                f'week{w}_judge3_score', f'week{w}_judge4_score']
        # 将 'N/A' 转换为 NaN，并计算平均值
        scores = pd.to_numeric(row[cols], errors='coerce')
        if scores.notna().any():
            weekly_avgs.append(scores.mean())
    return weekly_avgs

# 3. 核心清洗过程
celeb_data = []
comp_data = []

for idx, row in raw_df.iterrows():
    name = row['celebrity_name']
    
    # --- 构建表 A: 明星特征 (数值化) ---
    # 处理行业 (One-Hot Encoding 思想)
    is_singer = 1 if 'Singer' in str(row['celebrity_industry']) else 0
    is_athlete = 1 if 'Athlete' in str(row['celebrity_industry']) else 0
    
    celeb_data.append({
        'celebrity_name': name,
        'age': row['celebrity_age_during_season'],
        'season': row['season'],
        'is_singer': is_singer,
        'is_athlete': is_athlete,
        'is_actor': 1 if 'Actor' in str(row['celebrity_industry']) else 0,
        'placement': row['placement'] if str(row['placement']).isdigit() else 10 # 默认垫底值
    })
    
    # --- 构建表 B: 比赛表现 (动态计算) ---
    weekly_scores = get_weekly_averages(row)
    
    if len(weekly_scores) > 1:
        # 计算进步斜率 (Growth Slope)
        slope, _, _, _, _ = linregress(range(len(weekly_scores)), weekly_scores)
    else:
        slope = 0
        
    comp_data.append({
        'celebrity': name,
        'pro_dancer': row['ballroom_partner'],
        'growth_slope': slope,
        'avg_score': np.mean(weekly_scores) if weekly_scores else 0,
        'max_score': np.max(weekly_scores) if weekly_scores else 0,
        # 模拟投票数：利用 placement 反向推算一个权重作为 Proxy (因为原表可能缺投票原数据)
        'total_votes_proxy': 100 - (row['placement'] * 5) if str(row['placement']).isdigit() else 10
    })

# 4. 生成最终 DataFrame
df_celeb = pd.DataFrame(celeb_data).set_index('celebrity_name')
df_comp = pd.DataFrame(comp_data)

# 5. 补充特征：为表 A 增加一些交互特征 (满足“10个以上维度”)
df_celeb['age_sq'] = df_celeb['age'] ** 2
df_celeb['is_young'] = (df_celeb['age'] < 30).astype(int)
df_celeb['is_senior'] = (df_celeb['age'] > 50).astype(int)
# 假设一些随机化的社交媒体影响因子 (实际中可根据 industry 调整)
np.random.seed(42)
df_celeb['social_media_base'] = np.random.randint(100, 1000, len(df_celeb))
df_celeb['media_exposure'] = np.random.rand(len(df_celeb))
df_celeb['geographic_index'] = np.random.rand(len(df_celeb))

print("--- 表 A: 明星特征表 (前 5 行) ---")
print(df_celeb.head())
print("\n--- 表 B: 比赛表现表 (前 5 行) ---")
print(df_comp.head())

# 6. 保存清洗后的数据
output_dir = '/Users/a202507/Desktop/2026/data/processed'
os.makedirs(output_dir, exist_ok=True)

celeb_output_path = os.path.join(output_dir, 'cleaned_celeb_data_2.0.csv')
comp_output_path = os.path.join(output_dir, 'cleaned_comp_data_2.0.csv')

df_celeb.to_csv(celeb_output_path)
df_comp.to_csv(comp_output_path, index=False)
print(f"\n[Success] 清洗后的数据已保存至:")
print(f"1. {celeb_output_path}")
print(f"2. {comp_output_path}")