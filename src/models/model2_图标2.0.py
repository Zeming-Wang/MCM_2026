import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# 统一学术样式
plt.rcParams['font.sans-serif'] = ['Arial']
sns.set_theme(style="whitegrid", palette="muted")

import os

# --- 输出路径设置 ---
output_dir = '/Users/a202507/Desktop/2026/data/processed/plots'
os.makedirs(output_dir, exist_ok=True)

# 1. 加载数据
data_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Data.csv'
df = pd.read_csv(data_path)

# --- 图表 1: 影响权重热图 (基于统计分析结果) ---
def plot_heatmap():
    plt.figure(figsize=(9, 6))
    data = {
        'Success Index': [0.139, -0.077, 0.183],
        'Growth Slope': [0.269, -0.062, -0.017]
    }
    heatmap_df = pd.DataFrame(data, index=['Pro Influence (IIS)', 'Physical Factor', 'Showbiz Factor'])
    sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn', center=0, cbar_kws={'label': 'Correlation Coefficient (r)'})
    plt.title("Figure 1: Factor Impact Heatmap", fontsize=15, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig1_Heatmap.png'), dpi=300)
    plt.show()

# --- 图表 2: 导师赋能回归图 (IIS vs. Growth Slope) ---
def plot_regression():
    plt.figure(figsize=(9, 6))
    # 按照导师聚合计算平均斜率，模拟 IIS 指数
    df['pro_iis_proxy'] = df.groupby('ballroom_partner')['growth_slope'].transform('mean')
    sns.regplot(data=df, x='pro_iis_proxy', y='growth_slope', 
                scatter_kws={'s': 80, 'alpha': 0.6, 'color': "#5ba3eb"}, 
                line_kws={'color': '#e74c3c', 'lw': 3})
    plt.title("Figure 2: Professional Mentorship Effect", fontsize=15)
    plt.xlabel("Pro Dancer Influence Index (IIS)", fontsize=12)
    plt.ylabel("Celebrity Growth Slope", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig2_Regression.png'), dpi=300)
    plt.show()

# --- 图表 3: 行业背景对比箱线图 (Industry vs. Success Index) ---
def plot_box_industry():
    plt.figure(figsize=(10, 6))
    # 逆向转换哑变量，方便绘图
    plot_df = df.copy()
    conditions = [
        (plot_df['is_athlete'] == 1),
        (plot_df['is_performer'] == 1),
        (plot_df['is_reality_social'] == 1)
    ]
    choices = ['Athlete', 'Performer', 'Social/Reality']
    plot_df['Industry_Group'] = np.select(conditions, choices, default='Other')
    
    sns.boxplot(data=plot_df, x='Industry_Group', y='success_index', palette="Set2", width=0.5)
    sns.stripplot(data=plot_df, x='Industry_Group', y='success_index', color=".3", size=5, alpha=0.5)
    plt.title("Figure 3: Success Distribution by Industry Background", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig3_Industry_Box.png'), dpi=300)
    plt.show()

# --- 图表 4: 学习轨迹对比图 (Learning Curves) ---
def plot_learning_curves():
    plt.figure(figsize=(11, 7))
    def parse_scores(s):
        if not isinstance(s, str): return []
        try:
            # 兼容带有 np.float64(...) 的字符串
            import re
            cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', s)
            return ast.literal_eval(cleaned)
        except: return []

    top_3 = df.nsmallest(3, 'placement')
    bottom_3 = df.nlargest(3, 'placement')

    for _, row in top_3.iterrows():
        scores = parse_scores(row['score_series'])
        if scores:
            plt.plot(range(1, len(scores)+1), scores, marker='o', lw=2, label=f"Finalist: {row['celebrity_name']}")
    
    for _, row in bottom_3.iterrows():
        scores = parse_scores(row['score_series'])
        if scores:
            plt.plot(range(1, len(scores)+1), scores, marker='x', ls='--', alpha=0.7, label=f"Early Out: {row['celebrity_name']}")

    plt.title("Figure 4: Skill Acquisition Trajectories", fontsize=15)
    plt.xlabel("Week", fontsize=12)
    plt.ylabel("Mean Score", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig4_Curves.png'), dpi=300)
    plt.show()

# 运行所有绘图函数
plot_heatmap()
plot_regression()
plot_box_industry()
plot_learning_curves()
print(f"所有图表已成功生成并保存至: {output_dir}")