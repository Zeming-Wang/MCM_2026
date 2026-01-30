#目的： 展示特征贡献度，并结合 MCMC 采样的不确定性宽度 。#
import numpy as np
import matplotlib.pyplot as plt
def plot_sensitivity_radar(sobol_indices, hdi_95):
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
    
    # 图注 (Caption): 
    # 雷达图量化了不同“软特征”对粉丝投票的影响权重。外围阴影区展示了 MCMC 采样
    # 的不确定性。当面临极端数据（Outliers）时，阴影区的扩张提醒决策者模型此时的预测依赖于更宽的概率分布。
