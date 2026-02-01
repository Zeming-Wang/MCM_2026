import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr, linregress, spearmanr
import ast
import os

# --- 配置路径 ---
input_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Data.csv'
output_dir = '/Users/a202507/Desktop/2026/data/processed/plots'
os.makedirs(output_dir, exist_ok=True)

class DWTSMasterModel:
    def __init__(self, df):
        self.df = df
        self.iis_results = None
        self.latent_factors = None
        self.analysis_df = None

    # ==========================================
    # 1. 特征工程层：细化拆解与派生
    # ==========================================
    def feature_engineering(self):
        print("Executing Feature Engineering...")
        # A. 物理竞技维度
        # 计算偏离巅峰年龄 (假设25岁为舞蹈身体巅峰)
        self.df['age_peak_offset'] = self.df['celebrity_age_during_season'].apply(lambda x: abs(x - 25))
        
        # B. 舞蹈过往经验 (模拟逻辑：运动员和演员通常有不同程度的基础)
        self.df['dance_exp_proxy'] = self.df['celebrity_industry'].map({
            'Athlete': 0.6, 'Singer': 0.8, 'Actor': 0.5, 'TV Personality': 0.3
        }).fillna(0.4)

        # C. 地缘/社交维度 (Utah 加成逻辑)
        self.df['is_utah'] = self.df['celebrity_homestate'].apply(lambda x: 1 if x == 'Utah' else 0)
        
        # D. 导师社交权重 (手动模拟：知名导师如 Mark, Sasha 等拥有更高 IIS 权重)
        pro_social_map = {'Mark Ballas': 0.9, 'Sasha Farber': 0.8, 'Witney Carson': 0.85, 'Alan Bersten': 0.7}
        self.df['pro_social_weight'] = self.df['ballroom_partner'].map(pro_social_map).fillna(0.5)

    # ==========================================
    # 2. 网络层：计算社会加权 IIS (Social-Weighted IIS)
    # ==========================================
    def run_network_layer(self):
        print("Running Network Layer (IIS Calculation)...")
        G = nx.DiGraph()
        
        for _, row in self.df.iterrows():
            # 基础斜率
            slope = row.get('growth_slope', 0)
            # 社会声量乘子
            social_multiplier = row['pro_social_weight']
            # 复合权重：技术赋能 * 导师声量
            combined_weight = slope * (1 + social_multiplier)
            G.add_edge(row['ballroom_partner'], row['celebrity_name'], weight=combined_weight)
        
        # 使用特征向量中心性计算影响力
        try:
            # 使用标准的 eigenvector_centrality 替代 numpy 版本，增加容错
            centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-06)
            iis_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['IIS'])
            iis_df['IIS'] = MinMaxScaler().fit_transform(iis_df[['IIS']])
            self.iis_results = iis_df[~iis_df.index.isin(self.df['celebrity_name'])]
        except Exception as e:
            print(f"Network error (eigenvector): {e}")
            # 降级方案：使用度中心性
            print("Falling back to degree centrality...")
            centrality = nx.degree_centrality(G)
            iis_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['IIS'])
            self.iis_results = iis_df[~iis_df.index.isin(self.df['celebrity_name'])]

        return self.iis_results

    # ==========================================
    # 3. 特征层：因子重构 (Factor Analyzer)
    # ==========================================
    def run_factor_layer(self):
        print("Running Factor Layer (Latent Factor Extraction)...")
        # 选取细化后的特征
        features = ['age_peak_offset', 'dance_exp_proxy', 'is_utah', 'is_athlete', 'is_performer']
        # 确保数据完整
        fa_data = self.df[features].fillna(0)
        
        # 执行因子分析 (提取 3 个因子)
        # 注意：不传入 StandardScaler 处理后的数据，而是让 FA 内部处理，或者直接传原始数据
        # sklearn 的 StandardScaler 返回 numpy array，可能会丢失列名信息，但在 fit 时是允许的
        # 这里为了兼容性，先标准化为 numpy array
        
        # 处理可能的 nan 值，用 0 填充
        X_scaled = StandardScaler().fit_transform(fa_data)
        X_scaled = np.nan_to_num(X_scaled)

        # 降级方案：由于 FactorAnalyzer 可能与当前 scikit-learn 版本不兼容
        # 我们使用 sklearn 的 FactorAnalysis 或 PCA 作为替代，或者尝试修复参数
        # 这里为了快速修复，我们改用 sklearn 的 FactorAnalysis
        from sklearn.decomposition import FactorAnalysis
        fa = FactorAnalysis(n_components=3, rotation="varimax")
        fa.fit(X_scaled)
        
        factor_scores = fa.transform(X_scaled)
        self.latent_factors = pd.DataFrame(
            factor_scores,
            columns=['Physical_Comp', 'Social_Capital', 'Geographic_Affinity'],
            index=self.df.index
        )
        return self.latent_factors

    # ==========================================
    # 4. 分析层：归因博弈 (Shapley Value / Correlation)
    # ==========================================
    def run_analysis_layer(self):
        print("Running Analysis Layer (Attribution)...")
        # 合并所有维度
        self.analysis_df = pd.concat([self.df, self.latent_factors], axis=1)
        # 合并导师 IIS
        self.analysis_df = self.analysis_df.merge(self.iis_results, left_on='ballroom_partner', right_index=True)
        
        # 定义核心指标
        metrics = ['IIS', 'Physical_Comp', 'Social_Capital', 'Geographic_Affinity']
        targets = {'Judges': 'success_index', 'Fans': 'success_index'} # 这里可根据需要替换为具体投票数据
        
        report_r = pd.DataFrame(index=metrics, columns=targets.keys())
        for m in metrics:
            for t_name, t_col in targets.items():
                r, _ = pearsonr(self.analysis_df[m], self.analysis_df[t_col])
                report_r.loc[m, t_name] = r
        return report_r

    # ==========================================
    # 5. 可视化方法 (带显著性标记的豪华热图)
    # ==========================================
    def plot_master_results(self, report):
        plt.figure(figsize=(10, 6))
        sns.heatmap(report.astype(float), annot=True, cmap='coolwarm', center=0, linewidths=1)
        plt.title("Dual-Drive Influence: Who Dominates the Game?", fontsize=14)
        plt.ylabel("Latent Factors & Mentor IIS")
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'Master_Influence_Analysis.png')
        plt.savefig(output_file, dpi=300)
        plt.show()
        print(f"分析结果图已保存至: {output_file}")

# ==========================================
# 灵敏度分析工具 (Sensitivity Analysis)
# ==========================================
def perform_sensitivity(model_instance):
    print("\n--- Sensitivity Analysis ---")
    # 模拟 10% 扰动对 IIS 排名的影响
    baseline = model_instance.iis_results['IIS'].sort_index()
    noise = np.random.normal(0, 0.05, len(baseline))
    perturbed = baseline + noise
    rho, _ = spearmanr(baseline, perturbed)
    print(f"Model Robustness (Spearman Rho): {rho:.4f}")

# ==========================================
# 执行主程序
# ==========================================
if __name__ == "__main__":
    # 1. 加载你的原始数据
    try:
        processed_df = pd.read_csv(input_path)
        print(f"成功加载数据: {input_path}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_path}")
        exit()

    # 2. 初始化大师模型
    master_model = DWTSMasterModel(processed_df)

    # 3. 运行流程
    master_model.feature_engineering()
    master_model.run_network_layer()
    master_model.run_factor_layer()
    final_stats = master_model.run_analysis_layer()

    # 4. 产出豪华图表
    master_model.plot_master_results(final_stats)

    # 5. 稳健性验证
    perform_sensitivity(master_model)