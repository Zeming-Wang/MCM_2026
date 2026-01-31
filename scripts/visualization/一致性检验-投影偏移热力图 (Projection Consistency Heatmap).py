#目的： 验证贝叶斯残差项是否在物理约束的可接受范围内，量化投影算子P对原始预测的校正强度 。#
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_projection_consistency(v_raw_list, v_projected_list, output_path=None):
    """
    v_raw_list: 融合但未投影的得分向量 (v_base + alpha * Delta_V)
    v_projected_list: 投影到单纯形后的最终得分 (V_final)
    """
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    
    # 绘制散点与热力密度
    sns.kdeplot(x=v_raw_list, y=v_projected_list, cmap="viridis", fill=True, alpha=0.3, ax=ax)
    ax.scatter(v_raw_list, v_projected_list, c=v_projected_list, cmap='viridis', 
               edgecolor='white', s=60, alpha=0.8, label='Samples')
    
    # 参考线 y=x
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal Consistency')
    
    ax.set_title("Figure 1: Projection Fidelity & Constraint Consistency", fontsize=12)
    ax.set_xlabel("Unconstrained Fusion Score ($V_{base} + \\alpha \\Delta V$)", fontsize=10)
    ax.set_ylabel("Final Projected Probability ($V_{final}$)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    charts_dir = r"d:\MCM_2026_O\charts"
    if output_path is None:
        output_path = os.path.join(charts_dir, "projection_consistency_heatmap.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    sample_v_raw = [0.1, 0.2, 0.3, 0.4, 0.5]
    sample_v_proj = [0.12, 0.21, 0.29, 0.39, 0.49]
    plot_projection_consistency(sample_v_raw, sample_v_proj)

# 图注 (Caption): 
# 该图展示了贝叶斯残差融合结果与物理约束空间的一致性。散点聚集在 y=x 近邻说明
# 模型在考虑粉丝偏好（Delta V）时并未严重偏离物理可行域。若散点出现大规模非线性偏移，
# 则表明物理约束（如淘汰规则）正在强制修正不合理的统计预测。
