import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
import os

# --- 配置路径 ---
input_path = '/Users/a202507/Desktop/2026/data/processed/model2_influence_results.csv'
output_dir = '/Users/a202507/Desktop/2026/data/processed/plots'
os.makedirs(output_dir, exist_ok=True)

# 1. 读取数据 (请确保文件名一致)
try:
    df_results = pd.read_csv(input_path, index_col=0)
    print(f"成功读取数据: {input_path}")
except FileNotFoundError:
    print(f"错误: 找不到文件 {input_path}")
    exit()

# --- 核心函数：从字符串中提取 r 值和 p 值 ---
def extract_stats(text):
    r_val = float(re.search(r'r=(-?\d+\.\d+)', text).group(1))
    p_val = float(re.search(r'p=(\d+\.\d+)', text).group(1))
    return r_val, p_val

# 解析数据
r_matrix = df_results.applymap(lambda x: extract_stats(x)[0])
p_matrix = df_results.applymap(lambda x: extract_stats(x)[1])

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial']
sns.set_theme(style="white")

# --- 图 1: 相关性热图 (Correlation Heatmap) ---
plt.figure(figsize=(10, 6))
# 显著性标注：如果 p < 0.05，在格子里加个星号 *
annot_labels = r_matrix.copy().astype(str)
for i in range(r_matrix.shape[0]):
    for j in range(r_matrix.shape[1]):
        val = r_matrix.iloc[i, j]
        p = p_matrix.iloc[i, j]
        star = "*" if p < 0.05 else ""
        annot_labels.iloc[i, j] = f"{val:.3f}{star}"

sns.heatmap(r_matrix, annot=annot_labels, fmt="", cmap='RdYlGn', center=0, 
            linewidths=1, cbar_kws={'label': 'Correlation Coefficient (r)'})
plt.title("Figure 1: Impact Factor Correlation Matrix\n(* indicates p < 0.05)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Chart_Heatmap.png'), dpi=300)
plt.show()

# --- 图 2: 影响权重对比条形图 (Factor Importance Bar Chart) ---
# 重点展示对 Success Index (最终名次) 的贡献
plt.figure(figsize=(10, 6))
success_impact = r_matrix['success_index'].sort_values(ascending=False)
colors = ["#f0b205" if x > 0 else '#e74c3c' for x in success_impact]

success_impact.plot(kind='barh', color=colors, alpha=0.8)
plt.axvline(0, color='black', lw=1)
plt.title("Figure 2: Relative Contribution to Success Index", fontsize=14)
plt.xlabel("Correlation Strength (r)")
plt.ylabel("Influence Factors")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Chart_Factor_Importance.png'), dpi=300)
plt.show()

print(f"两张图表已生成并保存至: {output_dir}")