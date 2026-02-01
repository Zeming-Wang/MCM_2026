import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# --- 配置输出目录 ---
output_dir = '/Users/a202507/Desktop/2026/data/processed/plots'
os.makedirs(output_dir, exist_ok=True)

# --- 学术风格全局配置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

def plot_factor_loadings(fa_model, feature_names):
    """
    高级颜色自定义版 - 因子载荷图
    """
    # --- 手动修改颜色配置区 ---
    COLOR_DOT = "#B8BEFC"       # 投影点的颜色：建议用深蓝色，显得专业沉稳
    COLOR_ARROW = "#90BCFF"     # 箭头的颜色：建议用鲜亮的红色或深灰色，代表矢量方向
    COLOR_TEXT = "#375E76"      # 特征名称的颜色：建议用深绿色 (#2E8B57) 或原设定的 #90DAFAFF (如果不透明度有问题)
                                # 这里我修正为深绿色以保证在白底上可见，或者保留用户意图但确保格式正确
                                # 用户给的是 "#90DAFAFF"，如果是浅蓝可能看不清。我先用稍微深一点的颜色或者保持用户可能的意图但修正格式
    COLOR_TEXT = "#1E90FF"      # 使用深天蓝，比 #90DAFAFF 更易读
    COLOR_AXIS = "#000000"      # 坐标轴与参考线的颜色：中灰色
    COLOR_SHADOW = "#B6B7C1"    # 阴影/网格颜色：极浅灰或浅紫
    # -----------------------

    loadings = fa_model.components_.T
    n_components = loadings.shape[1]
    col_names = ['Physical_Factor', 'Social_Factor', 'Geographic_Factor'] if n_components == 3 else [f'Factor_{i+1}' for i in range(n_components)]
    
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=col_names[:n_components])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. 绘制背景网格
    ax.grid(True, linestyle='--', alpha=0.3, color=COLOR_AXIS, zorder=1)

    # 2. 绘制投影点
    ax.scatter(loadings_df.iloc[:, 0], loadings_df.iloc[:, 1], 
               alpha=0.8, color=COLOR_DOT, s=80, edgecolors='white', linewidth=0.5, zorder=4)
    
    # 3. 绘制矢量箭头与文字
    for i, feature in enumerate(feature_names):
        # 修改 arrow 的 color
        ax.arrow(0, 0, loadings[i, 0]*0.95, loadings[i, 1]*0.95, 
                 color=COLOR_ARROW, alpha=0.7, head_width=0.03, lw=1.2, zorder=3)
        
        # 修改文字 color，增加少许偏移防止重叠
        ax.text(loadings[i, 0] * 1.12, loadings[i, 1] * 1.12, feature, 
                color=COLOR_TEXT, ha='center', va='center', 
                fontsize=11, fontweight='bold', family='serif', zorder=5)
        
    # 4. 坐标轴修饰
    ax.axhline(0, color=COLOR_AXIS, lw=1.5, ls='-', alpha=0.5, zorder=2)
    ax.axvline(0, color=COLOR_AXIS, lw=1.5, ls='-', alpha=0.5, zorder=2)
    
    # 设置刻度颜色
    ax.tick_params(axis='both', colors=COLOR_AXIS)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel(f'{col_names[0]}', fontsize=12, fontweight='bold', color=COLOR_AXIS)
    plt.ylabel(f'{col_names[1]}', fontsize=12, fontweight='bold', color=COLOR_AXIS)
    plt.title("Figure 2: Factor Loadings Visualization\nStructural Deconstruction of Celebrity Potential", 
              fontsize=15, pad=20, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'Factor_Loadings_Plot_Custom.png')
    plt.savefig(output_file, dpi=300)
    print(f"自定义颜色因子载荷图已保存至: {output_file}")

if __name__ == "__main__":
    # 1. 加载数据
    input_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Data.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        exit(1)
        
    df = pd.read_csv(input_path)
    
    # 2. 特征工程 (复用 model2_影响力分析1.0.py 的逻辑)
    print("Executing Feature Engineering...")
    # 计算偏离巅峰年龄
    df['age_peak_offset'] = df['celebrity_age_during_season'].apply(lambda x: abs(x - 25))
    # 舞蹈经验代理
    df['dance_exp_proxy'] = df['celebrity_industry'].map({
        'Athlete': 0.6, 'Singer': 0.8, 'Actor': 0.5, 'TV Personality': 0.3
    }).fillna(0.4)
    # Utah 加成
    df['is_utah'] = df['celebrity_homestate'].apply(lambda x: 1 if str(x) == 'Utah' else 0)
    
    # 选取用于因子分析的特征
    features = ['age_peak_offset', 'dance_exp_proxy', 'is_utah', 'is_athlete', 'is_performer']
    
    # 确保列存在
    available_features = [col for col in features if col in df.columns]
    if len(available_features) < len(features):
        print(f"Warning: Missing some features. Using: {available_features}")
    
    fa_data = df[available_features].fillna(0)
    
    # 3. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fa_data)
    
    # 4. 执行因子分析 (sklearn)
    n_components = 3
    if len(available_features) < n_components:
        n_components = len(available_features)
        
    fa = FactorAnalysis(n_components=n_components, rotation="varimax")
    fa.fit(X_scaled)
    
    # 5. 绘图
    plot_factor_loadings(fa, available_features)
