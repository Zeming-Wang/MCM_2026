import pandas as pd
import numpy as np
import networkx as nx
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import linregress

# ==========================================
# 1. 增强型数据模拟 (包含社交声量与多维特征)
# ==========================================
def generate_advanced_data(n_samples=40):
    np.random.seed(42)
    # 模拟明星特征 (10+ 变量)
    celeb_features = {
        'age': np.random.randint(18, 60, n_samples),
        'base_followers': np.random.randint(10, 1000, n_samples), # 存量名气
        'social_activity': np.random.rand(n_samples), # 社交活跃度
        'body_mass_index': np.random.normal(22, 3, n_samples), # 身体素质指标
        'dance_exp': np.random.randint(0, 5, n_samples),
        'geo_affinity': np.random.rand(n_samples), # 地缘亲和力
        'media_exposure': np.random.rand(n_samples),
        'charity_work': np.random.randint(0, 10, n_samples),
        'athletic_bg': np.random.choice([0, 1], n_samples), # 运动员背景
        'acting_skills': np.random.rand(n_samples)
    }
    df_celeb = pd.DataFrame(celeb_features, index=[f'Celeb_{i}' for i in range(n_samples)])

    # 模拟比赛过程
    pros = [f'Pro_{i}' for i in 'ABCDE']
    pro_social_power = {p: np.random.rand() for p in pros} # 导师社交影响力
    
    comp_data = []
    for celeb_id in df_celeb.index:
        pro_id = np.random.choice(pros)
        # 1. 技术得分演进 (模拟 8 周)
        weekly_scores = np.sort(np.random.normal(loc=25, scale=4, size=8))
        growth_slope = linregress(np.arange(8), weekly_scores).slope
        
        # 2. 粉丝投票 (受导师声量、选手名气、技术表现共同影响)
        votes = (df_celeb.loc[celeb_id, 'base_followers'] * 0.4 + 
                 pro_social_power[pro_id] * 500 + 
                 weekly_scores.mean() * 10 + 
                 np.random.normal(0, 50))
        
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
# 2. 核心架构：NFA 评估类
# ==========================================
class NFAModelFramework:
    def __init__(self, df_celeb, df_comp):
        self.df_celeb = df_celeb
        self.df_comp = df_comp
        self.iis_scores = None
        self.latent_factors = None

    # --- 第一阶段：网络层 (IIS 计算) ---
    def compute_network_iis(self):
        G = nx.DiGraph()
        for _, row in self.df_comp.iterrows():
            # 边权 = 技术赋能 (W_tech) + 声量外溢 (W_social)
            weight = row['growth_slope'] * 0.7 + row['pro_social_spillover'] * 0.3
            G.add_edge(row['pro_dancer'], row['celebrity'], weight=weight)
        
        # 特征向量中心性识别核心影响节点
        # 修正：处理不连通图的情况，或者使用标准 eigenvector_centrality
        try:
            # 尝试使用标准方法，容忍度设大一点
            centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-06)
        except:
            # 如果失败 (例如不连通或不收敛)，回退到度中心性
            print("Warning: Eigenvector centrality failed (possibly disconnected graph). Falling back to Degree Centrality.")
            centrality = nx.degree_centrality(G)
            
        iis_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['IIS'])
        self.iis_scores = iis_df[iis_df.index.str.startswith('Pro_')]
        return self.iis_scores

    # --- 第二阶段：特征层 (因子提炼) ---
    def extract_latent_factors(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df_celeb)
        
        # 提取 3 个潜因子：身体竞争力、存量名气、地缘亲和力
        # 降级方案：由于 FactorAnalyzer 可能与当前 scikit-learn 版本不兼容
        # 我们使用 sklearn 的 FactorAnalysis
        from sklearn.decomposition import FactorAnalysis
        fa = FactorAnalysis(n_components=3, rotation="varimax")
        fa.fit(scaled_data)
        
        factor_names = ['Physique_Factor', 'Popularity_Factor', 'Affinity_Factor']
        scores = fa.transform(scaled_data)
        self.latent_factors = pd.DataFrame(scores, columns=factor_names, index=self.df_celeb.index)
        return self.latent_factors

    # --- 第三阶段：分析层 (Shapley Value 归因) ---
    def run_attribution_analysis(self):
        # 合并所有特征
        full_df = self.df_comp.merge(self.iis_scores, left_on='pro_dancer', right_index=True)
        full_df = full_df.merge(self.latent_factors, left_on='celebrity', right_index=True)
        
        features = ['IIS', 'Physique_Factor', 'Popularity_Factor', 'Affinity_Factor']
        
        # 针对“总投票数”构建回归模型
        X = full_df[features]
        y = full_df['total_votes']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 计算 Shapley Value
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        return X, shap_values, full_df

# ==========================================
# 3. 运行与结果展示
# ==========================================
import os

# ... (previous imports)

# ==========================================
# 3. 运行与结果展示
# ==========================================
# 初始化数据与模型
df_celeb, df_comp = generate_advanced_data()
nfa = NFAModelFramework(df_celeb, df_comp)

# 执行层级分析
iis_results = nfa.compute_network_iis()
factors = nfa.extract_latent_factors()
X_attr, shap_vals, final_data = nfa.run_attribution_analysis()

# --- 结果输出与保存 ---
output_dir = '/Users/a202507/Desktop/2026/results'
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# 1. 导师 IIS 排行
print("--- 导师影响力 IIS 评估 ---")
iis_sorted = iis_results.sort_values(by='IIS', ascending=False)
print(iis_sorted)
iis_csv_path = os.path.join(output_dir, 'model2_IIS_rankings.csv')
iis_sorted.to_csv(iis_csv_path)
print(f"结果已保存至: {iis_csv_path}")

# 2. Shapley 归因摘要
plt.figure(figsize=(10, 6))
plt.title("Shapley Value Attribution: Contribution to Total Votes")
shap.summary_plot(shap_vals, X_attr, plot_type="bar", show=False)
shap_plot_path = os.path.join(plots_dir, 'model2_shapley_attribution.png')
plt.savefig(shap_plot_path)
print(f"图表已保存至: {shap_plot_path}")
plt.show()

# 3. 双路径关联热图
plt.figure(figsize=(8, 5))
correlation = final_data[['avg_score', 'total_votes', 'IIS', 'Physique_Factor', 'Popularity_Factor']].corr()
sns.heatmap(correlation[['avg_score', 'total_votes']].drop(['avg_score', 'total_votes']), 
            annot=True, cmap='coolwarm')
plt.title("Dual-Path Correlation: Judges vs Fans")
heatmap_path = os.path.join(plots_dir, 'model2_dual_path_correlation.png')
plt.savefig(heatmap_path)
print(f"图表已保存至: {heatmap_path}")
plt.show()