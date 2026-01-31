#目的： 证明模型生成的预测结果在任何规则下（排名或百分比）均能导致正确的淘汰结果 。#
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import os

def plot_boundary_fidelity(df_results, output_path=None):
    """
    df_results: 包含列 ['JudgeRank', 'PredictedScoreRank', 'EliminatedStatus']
    """
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 使用 viridis 颜色区分淘汰状态
    color_map = {'Eliminated': "#AC23CF8B", 'Safe': "#0A9F05AE", 'Winner': '#FDE725'}
    
    parallel_coordinates(df_results, 'EliminatedStatus', color=("#F1B8E9B9", "#5F66E6"), alpha=0.6)
    
    plt.gca().invert_yaxis() # 排名越小越好
    plt.title("Figure 2: Multi-layer Rank Consistency & Elimination Fidelity", fontsize=12)
    plt.ylabel("Rank Order (1st is Top)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    charts_dir = r"d:\MCM_2026_O\charts"
    if output_path is None:
        output_path = os.path.join(charts_dir, "boundary_fidelity.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    data = {
        "JudgeRank": [1, 2, 3],
        "PredictedScoreRank": [1, 3, 2],
        "EliminatedStatus": ["Winner", "Eliminated", "Safe"],
    }
    df_example = pd.DataFrame(data)
    plot_boundary_fidelity(df_example)
