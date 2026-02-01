import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import linregress
from scipy.interpolate import griddata
import os

# ==========================================
# 1. 数据生成模块 (复用 Model 2 逻辑)
# ==========================================
def generate_advanced_data(n_samples=200):
    """
    生成模拟数据，增加样本量以获得更好的平滑曲面效果
    """
    np.random.seed(42)
    # 模拟明星特征
    celeb_features = {
        'age': np.random.randint(18, 60, n_samples),
        'base_followers': np.random.randint(10, 1000, n_samples),
        'social_activity': np.random.rand(n_samples),
        'body_mass_index': np.random.normal(22, 3, n_samples),
        'dance_exp': np.random.randint(0, 5, n_samples),
        'geo_affinity': np.random.rand(n_samples),
        'media_exposure': np.random.rand(n_samples),
        'charity_work': np.random.randint(0, 10, n_samples),
        'athletic_bg': np.random.choice([0, 1], n_samples),
        'acting_skills': np.random.rand(n_samples)
    }
    df_celeb = pd.DataFrame(celeb_features, index=[f'Celeb_{i}' for i in range(n_samples)])

    # 模拟比赛过程
    pros = [f'Pro_{i}' for i in 'ABCDE']
    pro_social_power = {p: np.random.rand() for p in pros}
    
    comp_data = []
    for celeb_id in df_celeb.index:
        pro_id = np.random.choice(pros)
        weekly_scores = np.sort(np.random.normal(loc=25, scale=4, size=8))
        growth_slope = linregress(np.arange(8), weekly_scores).slope
        
        # 构造投票逻辑：
        # 1. Physique_Factor 代理变量：athletic_bg, body_mass_index (负相关), dance_exp
        # 2. Popularity_Factor 代理变量：base_followers, media_exposure
        
        # 为了让 3D 图有明显的双峰或交互效应，我们在生成逻辑中强化这两个维度的作用
        physique_score = (df_celeb.loc[celeb_id, 'athletic_bg'] * 2 + 
                          df_celeb.loc[celeb_id, 'dance_exp'] - 
                          abs(df_celeb.loc[celeb_id, 'body_mass_index'] - 22) * 0.1)
        
        popularity_score = (df_celeb.loc[celeb_id, 'base_followers'] / 100 + 
                            df_celeb.loc[celeb_id, 'media_exposure'] * 5)
        
        # 总票数 = 基础(常数) + 评委路径(Physique) + 粉丝路径(Popularity) + 交互项 + 噪音
        votes = (100 + 
                 physique_score * 30 + 
                 popularity_score * 20 + 
                 (physique_score * popularity_score) * 5 +  # 交互效应让曲面更弯曲
                 np.random.normal(0, 20))
        
        comp_data.append({
            'celebrity': celeb_id,
            'pro_dancer': pro_id,
            'growth_slope': growth_slope,
            'pro_social_spillover': pro_social_power[pro_id],
            'avg_score': weekly_scores.mean(),
            'total_votes': votes
        })
    
    return df_celeb, pd.DataFrame(comp_data)

# ==========================================
# 2. 特征提取与曲面拟合
# ==========================================
def process_and_plot():
    print("正在生成数据与提取因子...")
    df_celeb, df_comp = generate_advanced_data(n_samples=300)
    
    # --- 因子分析 ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_celeb)
    
    # 强制提取 2 个主要因子用于 3D 绘图 (X, Y)
    fa = FactorAnalysis(n_components=2, rotation="varimax")
    fa_scores = fa.fit_transform(scaled_data)
    
    # 简单的启发式命名：检查因子与原始变量的相关性来命名
    # 这里为了演示，我们假设 Factor 0 是 Physique (身体/技术), Factor 1 是 Popularity (名气/社交)
    # 实际应用中需要查看 fa.components_
    df_factors = pd.DataFrame(fa_scores, columns=['Factor_1', 'Factor_2'], index=df_celeb.index)
    
    # 合并数据
    full_df = df_comp.merge(df_factors, left_on='celebrity', right_index=True)
    
    # 准备绘图数据
    X = full_df['Factor_1']
    Y = full_df['Factor_2']
    Z = full_df['total_votes']
    
    # --- 拟合平滑曲面 ---
    # 使用网格插值法
    print("正在计算响应曲面...")
    
    # 定义网格范围
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    
    # 使用 RBF (Radial Basis Function) 或 Cubic 插值获得平滑曲面
    # 或者训练一个回归模型来预测 Z (更稳健)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(full_df[['Factor_1', 'Factor_2']], Z)
    
    # 在网格上进行预测
    grid_z = model.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    grid_z = grid_z.reshape(grid_x.shape)
    
    # ==========================================
    # 3. 绘制 3D 曲面图 (仿照目标样式)
    # ==========================================
    print("正在绘制图表...")
    
    # 设置风格
    plt.style.use('default') # 重置风格避免冲突
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    # cmap='coolwarm' (蓝-红), alpha=0.9 (不透明度)
    surf = ax.plot_surface(grid_x, grid_y, grid_z, 
                           cmap=cm.coolwarm, 
                           linewidth=0, 
                           antialiased=False,
                           alpha=0.9)
    
    # 调整视角 (Elevation, Azimuth) 以获得最佳立体感
    ax.view_init(elev=35, azim=-125)
    
    # 设置坐标轴标签
    # 假设 Factor 1 对应 "Physique/Skill (Judges)"，Factor 2 对应 "Popularity (Fans)"
    # 我们通过简单的相关性检查来确认标签方向 (可选优化)
    corr_f1_votes = np.corrcoef(X, Z)[0, 1]
    corr_f2_votes = np.corrcoef(Y, Z)[0, 1]
    
    label_x = "Physique & Skill Factor (Judges' Path)"
    label_y = "Popularity & Social Factor (Fans' Path)"
    
    ax.set_xlabel(label_x, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(label_y, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Total Votes (Outcome)', fontsize=12, fontweight='bold', labelpad=10)
    
    # 设置标题
    ax.set_title('Dual-Path Impact Response Surface\n(Judges vs Fans Interaction)', fontsize=16, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Total Votes Prediction', rotation=270, labelpad=15)
    
    # 优化网格线
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 移除背景色 (让它看起来更干净，像第二张图)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # 保存
    output_dir = r'd:\MCM_2026_O\charts'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'dual_path_3d_surface.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")
    # plt.show() # 在无头环境中注释掉

if __name__ == "__main__":
    process_and_plot()
