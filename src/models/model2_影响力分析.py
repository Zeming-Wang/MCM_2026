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
# 假设你已经准备好了数据 df_comp (比赛结果) 和 df_celeb (明星特征)
# iis_results = calculate_instructor_influence(df_comp)
# factor_results, loadings = extract_celebrity_factors(df_celeb)
# final_report = perform_impact_analysis(final_merged_df)