#目的： 展示特征贡献度，并结合 MCMC 采样的不确定性宽度 。#
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_sensitivity_radar(sobol_indices, hdi_95, output_path=None):
    """
    sobol_indices: 各特征的贡献度
    hdi_95: 95% 高密度区间宽度
    """
    categories = ['Industry', 'Age', 'Fan Base', 'Judge Offset', 'Volatility']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    values = list(sobol_indices) + [sobol_indices[0]]
    error = list(hdi_95) + [hdi_95[0]]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=150)
    
    # 绘制灵敏度主线
    ax.plot(angles, values, color="#7A70B5", linewidth=2, label='Sobol Sensitivity')
    ax.fill(angles, values, color='#839DD1', alpha=0.25)
    
    # 绘制 95% HDI 阴影区间
    upper = [v + e for v, e in zip(values, error)]
    lower = [v - e for v, e in zip(values, error)]
    ax.fill_between(angles, lower, upper, color='#F1766D', alpha=0.1, label='95% HDI (Uncertainty)')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title("Figure 3: Global Sensitivity Radar & Posterior Uncertainty", pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    charts_dir = r"d:\MCM_2026_O\charts"
    if output_path is None:
        output_path = os.path.join(charts_dir, "uncertainty_radar.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    example_sobol = [0.4, 0.3, 0.2, 0.15, 0.1]
    example_hdi = [0.05, 0.04, 0.03, 0.02, 0.01]
    plot_sensitivity_radar(example_sobol, example_hdi)
