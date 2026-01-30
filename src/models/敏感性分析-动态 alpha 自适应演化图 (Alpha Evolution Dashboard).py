#目的： 展示模型如何根据比赛阶段自动平衡“规则”与“统计数据” 。#
import matplotlib.pyplot as plt
def plot_alpha_evolution(weeks, alpha_vals, entropy_vals):
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
    
    # 图注 (Caption): 
    # 该图揭示了模型权重的演化逻辑：随着赛季推进选手减少，解空间熵 $H_c$ 逐渐下降（约束增强），
    # 权重 $\alpha$ 随之平滑调整。这证明了模型在高确定性阶段（后期）更尊重物理排名规则，
    # 而在混乱的早期阶段更依赖贝叶斯特征提取。
