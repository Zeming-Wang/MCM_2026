import pandas as pd
import numpy as np
import networkx as nx
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr

# ==========================================
# 第一阶段：网络层 - 导师影响力指数 (IIS)
# ==========================================
def calculate_instructor_influence(competition_data):
    """
    competition_data: DataFrame 包含 ['pro_dancer', 'celebrity', 'score_improvement']
    """
    G = nx.DiGraph()
    
    # 构建有向图，权重为得分涨幅
    for _, row in competition_data.iterrows():
        G.add_edge(row['pro_dancer'], row['celebrity'], weight=row['score_improvement'])
    
    # 1. 计算加权出度 (Weighted Out-degree)
    out_degree = dict(G.out_degree(weight='weight'))
    
    # 2. 计算特征向量中心性 (Eigenvector Centrality)
    # 捕捉“带出的明星是否也是大咖”
    try:
        eigen_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    except:
        eigen_centrality = {node: 0 for node in G.nodes()}

    # 3. 归一化并合成 导师影响力指数 (IIS)
    iis_df = pd.DataFrame({'out_degree': out_degree, 'eigen': eigen_centrality})
    scaler = MinMaxScaler()
    iis_scaled = scaler.fit_transform(iis_df)
    iis_df['IIS'] = iis_scaled.mean(axis=1) # 综合指标
    
    return iis_df[iis_df.index.isin(competition_data['pro_dancer'].unique())]

# ==========================================
# 第二阶段：特征层 - 明星潜力因子提取
# ==========================================
def extract_celebrity_factors(celebrity_features):
    """
    celebrity_features: DataFrame 包含 ['age', 'social_media_score', 'is_athlete', 'is_actor'...]
    """
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(celebrity_features)
    
    # 因子分析 - 提取 3 个潜因子
    fa = FactorAnalyzer(n_factors=3, rotation="varimax")
    fa.fit(scaled_data)
    
    # 获取因子得分 (Physicality, Public_Base, Artistry)
    factor_scores = fa.transform(scaled_data)
    factor_df = pd.DataFrame(factor_scores, 
                             columns=['Physicality', 'Public_Base', 'Artistry'],
                             index=celebrity_features.index)
    return factor_df, fa.loadings_

# ==========================================
# 第三阶段：分析层 - 评委 vs 粉丝 因果判定
# ==========================================
def perform_impact_analysis(merged_data):
    """
    merged_data: 整合了 IIS 指数、明星潜因子、评委评分、粉丝投票的总表
    """
    results = {}
    
    # 定义自变量：舞者影响力、明星身体机能、明星群众基础
    independent_vars = ['IIS', 'Physicality', 'Public_Base', 'Artistry']
    dependent_vars = ['judges_score', 'fan_votes']
    
    for dv in dependent_vars:
        results[dv] = {}
        for iv in independent_vars:
            # 计算皮尔逊相关系数
            corr, p_value = pearsonr(merged_data[iv], merged_data[dv])
            results[dv][iv] = {'correlation': corr, 'p_value': p_value}
            
    return pd.DataFrame(results)

# ==========================================
# 模拟执行代码
# ==========================================
if __name__ == "__main__":
    # 1. 加载数据
    data_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Data.csv'
    df = pd.read_csv(data_path)
    
    print("数据加载成功，开始分析...")

    # 2. 计算导师影响力 (IIS)
    # 构造 competition_data: 需要 pro_dancer, celebrity, score_improvement (使用 growth_slope 代替)
    comp_data = df[['ballroom_partner', 'celebrity_name', 'growth_slope']].copy()
    comp_data.columns = ['pro_dancer', 'celebrity', 'score_improvement']
    # 过滤无效数据
    comp_data = comp_data.dropna()
    
    iis_results = calculate_instructor_influence(comp_data)
    print("\n--- 导师影响力指数 (Top 5) ---")
    print(iis_results.sort_values('IIS', ascending=False).head(5))
    
    from sklearn.decomposition import PCA

    # 3. 提取明星潜因子
    # 使用清洗好的特征: age_norm, is_athlete, is_performer, is_reality_social
    feature_cols = ['age_norm', 'is_athlete', 'is_performer', 'is_reality_social']
    celeb_features = df[feature_cols].dropna()
    
    print("\n--- 开始因子分析 (使用 PCA) ---")
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(celeb_features)
        
        # 使用 PCA 替代 FactorAnalyzer 以提高兼容性
        pca = PCA(n_components=2)
        factor_scores = pca.fit_transform(scaled_data)
        
        # 命名因子 (假设)
        factor_df = pd.DataFrame(factor_scores, 
                                columns=['Factor_Physical', 'Factor_Showbiz'],
                                index=celeb_features.index)
        print("因子分析(PCA)完成。")
        print("解释方差比:", pca.explained_variance_ratio_)
        
    except Exception as e:
        print(f"因子分析跳过: {e}")
        # 创建全0的占位符，防止后续报错
        factor_df = pd.DataFrame(0, index=celeb_features.index, columns=['Factor_Physical', 'Factor_Showbiz'])

    # 4. 整合数据进行回归/相关性分析
    # 将 IIS 和 因子得分 合并回原数据
    
    # 映射 IIS 回原表
    df['IIS'] = df['ballroom_partner'].map(iis_results['IIS'])
    
    # 合并因子得分 (按索引)
    df = df.join(factor_df)
    
    # 准备最终分析表
    # 自变量: IIS, Factor_Physical, Factor_Showbiz
    # 因变量: success_index (作为 fan/judge 综合结果代理), growth_slope
    
    analysis_df = df[['IIS', 'Factor_Physical', 'Factor_Showbiz', 'success_index', 'growth_slope']].dropna()
    
    print("\n--- 最终影响分析 (Pearson Correlation) ---")
    results = {}
    ivs = ['IIS', 'Factor_Physical', 'Factor_Showbiz']
    dvs = ['success_index', 'growth_slope']
    
    for dv in dvs:
        results[dv] = {}
        for iv in ivs:
            if iv in analysis_df.columns and dv in analysis_df.columns:
                corr, p = pearsonr(analysis_df[iv], analysis_df[dv])
                results[dv][iv] = f"r={corr:.3f} (p={p:.3f})"
    
    res_df = pd.DataFrame(results)
    print(res_df)
    
    # 保存结果
    output_path = '/Users/a202507/Desktop/2026/data/processed/model2_influence_results.csv'
    res_df.to_csv(output_path)
    print(f"\n分析结果已保存至: {output_path}")