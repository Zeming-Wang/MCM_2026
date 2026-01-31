import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# 设置中文字体（如果是中文环境）或美化样式
plt.style.use('seaborn-v0_8-muted')
sns.set_context("talk")

import os
# --- 输出路径设置 ---
output_dir = '/Users/a202507/Desktop/2026/data/processed/'
os.makedirs(output_dir, exist_ok=True)

# 1. 加载数据
data_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Data.csv'
df = pd.read_csv(data_path)

# --- 图表 1: 核心相关性热图 (Heatmap) ---
# 这张图直接展示你的统计结果
plt.figure(figsize=(10, 6))
# 模拟你 model2 结果中的相关性数值进行展示
corr_data = {
    'Success_Index': [0.139, -0.077, 0.183],
    'Growth_Slope': [0.269, -0.062, -0.017]
}
corr_df = pd.DataFrame(corr_data, index=['Professional_Influence (IIS)', 'Physical_Factor', 'Showbiz_Factor'])

sns.heatmap(corr_df, annot=True, cmap='RdYlGn', center=0, fmt=".3f")
plt.title("Key Impact Factors: Scores vs. Success")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'impact_heatmap.png'))
plt.show()

# --- 图表 2: 导师影响力与进步速度散点图 (Regression) ---
plt.figure(figsize=(10, 6))
# 假设我们用一个简单的代理指标代表导师IIS（如按导师聚合的平均斜率）
pro_rank = df.groupby('ballroom_partner')['growth_slope'].transform('mean')
sns.regplot(x=pro_rank, y=df['growth_slope'], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.xlabel("Pro Dancer Influence Index (Aggregated)")
plt.ylabel("Celebrity Growth Slope (Improvement)")
plt.title("The 'Teacher Effect': Pro Quality vs. Celeb Progress")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pro_influence_reg.png'))
plt.show()

# --- 图表 3: 学习曲线对比图 (Learning Curves) ---
plt.figure(figsize=(12, 7))

# 选取前 5 名和后 5 名展示差异
top_3 = df.nsmallest(3, 'placement')
bottom_3 = df.nlargest(3, 'placement')

import re

def parse_score_series(series_str):
    if not isinstance(series_str, str):
        return series_str
    # 移除 'np.float64(' 和 ')'
    cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', series_str)
    try:
        return ast.literal_eval(cleaned)
    except:
        return []

def plot_curves(subset, linestyle, label_prefix):
    for _, row in subset.iterrows():
        # 将字符串形式的列表转回数值列表
        scores = parse_score_series(row['score_series'])
        if scores:
             plt.plot(range(1, len(scores)+1), scores, marker='o', linestyle=linestyle, label=f"{label_prefix}: {row['celebrity_name']}")



plot_curves(top_3, '-', 'Finalists')
plot_curves(bottom_3, '--', 'Early Out')

plt.xlabel("Week")
plt.ylabel("Average Judge Score")
plt.title("Learning Curves: Top Performers vs. Early Eliminations")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
plt.show()

print(f"可视化图表已生成并保存至: {output_dir}")