#目的： 展示模型如何根据比赛阶段自动平衡“规则”与“统计数据” 。#
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_alpha_evolution(weeks, alpha_vals, entropy_vals, output_path=None):
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    
    # 绘制 Alpha 曲线
    color1 = '#899FB0'
    ax1.set_xlabel('Competition Week')
    ax1.set_ylabel('Adaptive Weight ($\\alpha$)', color=color1, fontsize=11)
    ax1.plot(weeks, alpha_vals, marker='o', color=color1, linewidth=2.5, label='$\\alpha$ (Data-Driven Intensity)')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 共享 X 轴绘制熵曲线
    ax2 = ax1.twinx()
    color2 = '#81B3A9'
    ax2.set_ylabel('Solution Space Entropy ($H_c$)', color=color2, fontsize=11)
    ax2.plot(weeks, entropy_vals, marker='s', linestyle='--', color=color2, alpha=0.8, label='Entropy (Constraints)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title("Figure 4: Dynamic Balancing of Physical Constraints vs. Bayesian Priors")
    fig.tight_layout()
    charts_dir = r"d:\MCM_2026_O\charts"
    if output_path is None:
        output_path = os.path.join(charts_dir, "alpha_evolution.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    example_weeks = list(range(1, 11))
    example_alpha = [0.3 + 0.05 * i for i in range(10)]
    example_entropy = [1.0 - 0.06 * i for i in range(10)]
    plot_alpha_evolution(example_weeks, example_alpha, example_entropy)
