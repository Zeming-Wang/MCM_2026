#逻辑： 直接回应题目要求的“确定性衡量”，展示各变量对结果波动的贡献。#
import matplotlib.pyplot as plt  # 用于绘图，对应plt
import numpy as np               # 用于数值计算，对应np
def plot_sensitivity_tornado():
    features = ['行业影响力', '年龄因子', '历史粉丝量', '评委评分偏离度']
    sobol_indices = [0.45, 0.12, 0.33, 0.08]
    uncertainty = [0.05, 0.02, 0.04, 0.01] # MCMC 后验宽度
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))
    
    # 绘制条形图
    bars = ax.barh(y_pos, sobol_indices, xerr=uncertainty, align='center', 
                   color='skyblue', edgecolor="#BC98EF", capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontfamily='SimHei')
    ax.invert_yaxis()  
    ax.set_xlabel('Sobol 全局灵敏度指数 (Total-order Index)')
    ax.set_title('特征灵敏度与确定性宽度 (Sensitivity & Uncertainty Analysis)')
    
    # 添加数值标签
    for i, v in enumerate(sobol_indices):
        ax.text(v + 0.01, i, f'{v:.2f}', color='black', va='center')

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
