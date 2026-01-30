import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import networkx as nx
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# ==========================================
# 1. 基础物理约束层 (Base Layer - Model A)
# 逻辑：利用 Norman Biggs 的空间过滤逻辑，确定最小可行解空间
# ==========================================

"""
这里目前实现的是一个“无知先验”（Uniform Distribution）
在实际的物理约束模型，应当计算满足相应不等式约束
R_j + R_f > R_eliminated 的解空间
"""

class PhysicalConstraintModel:
    def __init__(self, judges_scores, results):
        self.judges_scores = judges_scores
        self.results = results  # 包含谁被淘汰的信息

    def get_feasible_base_votes(self):
        """
        核心逻辑：通过不等式约束 R_j + R_f > R_eliminated 
        返回一个基础的预测向量 V_base (期望均值)
        """
        # 此处简化为均值计算，实际应结合你之前的解空间穷举代码
        n_contestants = len(self.judges_scores)
        v_base = np.ones(n_contestants) / n_contestants 
        return v_base

# ==========================================
# 2. 贝叶斯残差扰动层 (Residual Layer - Model B)
# 逻辑：利用 pgmpy/PyMC3 构建特征扰动 BN
# ==========================================

def build_bayesian_residual_model(industry_idx, age_data, fan_base_data):
    """
    构建贝叶斯网络：Industry/Age -> Preference/Volatility -> Score_Residual
    利用 V-Structure 捕捉不同特征间的竞争关系
    """
    with pm.Model() as residual_model:
        # --- 先验分布 (Priors) ---
        # 外部特征节点
        industry_effect = pm.Normal('Industry_Effect', mu=0, sigma=1, shape=len(np.unique(industry_idx)))
        age_slope = pm.Normal('Age_Slope', mu=0, sigma=1)
        """
        特征嵌入，将离散的索引变为连续的隐变量
        """
        # 中间隐变量 (Latent Variables)
        # Audience_Preference 受行业和初始粉丝量影响
        pref_mu = industry_effect[industry_idx] + age_slope * age_data
        audience_pref = pm.Normal('Audience_Preference', mu=pt.mean(pref_mu), sigma=1)
        
        # 目标残差变量 (Score_Residual)
        # 这里体现了从特征到残差的非线性映射
        delta_v = pm.Normal('Delta_V', mu=audience_pref, sigma=0.5, shape=len(age_data))
        
        return residual_model

# ==========================================
# 3. 异构融合架构 (Ensemble Engine)
# 逻辑：残差校准 + 投影算子 P
# ==========================================

class ResidualCalibrationEnsemble:
    def __init__(self, alpha_init=0.5):
        self.alpha = alpha_init  # 自适应权重因子

    def projection_operator_P(self, v_raw):
        
        #投影算子 P：将融合结果强制映射回符合规则的单纯形空间 (Sum=1, >=0)
        #并满足淘汰不等式约束
        
        # 使用 Softmax 保证和为 1，或使用纠偏矩阵
        exp_v = np.exp(v_raw)
        return exp_v / np.sum(exp_v)

    def adaptive_weight_logic(self, solution_space_entropy):
        
        #根据解空间稀疏度 (信息熵) 动态调整 alpha
        #解空间越小，物理约束越强，alpha 越小
        
        self.alpha = 1.0 / (1.0 + np.exp(-solution_space_entropy))
        return self.alpha

    def fuse(self, v_base, delta_v_samples):
        # 计算残差均值
        delta_v_mean = np.mean(delta_v_samples, axis=0)
        # 融合公式：V_final = P(V_base + alpha * Delta_V)
        v_final_raw = v_base + self.alpha * delta_v_mean
        return self.projection_operator_P(v_final_raw)

# ==========================================
# 4. 深度验证与灵敏度分析 (Sensitivity & Robustness)
# 逻辑：Sobol Indices & 扰动实验
# ==========================================

def perform_sobol_analysis(ensemble_model):
    
    #使用 SALib 进行全局灵敏度分析，量化各特征对 V_final 的贡献
    
    problem = {
        'num_vars': 3,
        'names': ['Industry', 'Age', 'Fan_Base'],
        'bounds': [[0, 1], [0, 1], [0, 1]]
    }
    param_values = saltelli.sample(problem, 1024)
    # 模拟模型输出
    Y = np.zeros([param_values.shape[0]])
    for i, X in enumerate(param_values):
        # 注入扰动并计算输出变化
        Y[i] = np.var(ensemble_model.projection_operator_P(X))
    
    Si = sobol.analyze(problem, Y)
    return Si # 返回 Sobol 指数

# ==========================================
# 5. 主逻辑执行流程
# ==========================================

if __name__ == "__main__":
    # --- 数据路径占位 ---
    DATA_PATH = "2026_MCM_Problem_C_Data.csv" 
    
    # 1. 初始化物理层 (模型 A)
    # scores = pd.read_csv(DATA_PATH)... (此处待后续处理)
    test_scores = np.array([28, 29, 30, 30])
    phys_model = PhysicalConstraintModel(test_scores, "Eliminated_Week_9")
    v_base = phys_model.get_feasible_base_votes()
    
    # 2. 运行贝叶斯残差层 (模型 B)
    # 构造模拟特征数据
    age_mock = np.array([24, 35, 29, 41])
    ind_mock = np.array([0, 1, 0, 2]) # 行业分类索引
    fb_mock = np.array([0.1, 0.4, 0.3, 0.2])
    
    res_model = build_bayesian_residual_model(ind_mock, age_mock, fb_mock)
    with res_model:
        # 进行最大后验估计 (MAP)
        map_estimate = pm.find_MAP()
        # 进行 MCMC 采样获取后验宽度 (Uncertainty)
        trace = pm.sample(500, tune=500, chains=2)
    
    # 3. 融合与校准
    ensemble = ResidualCalibrationEnsemble()
    # 计算解空间熵 (示例值)
    h_c = 0.44 / 0.720 # 引用 Norman Biggs 的例子比率
    alpha = ensemble.adaptive_weight_logic(h_c)
    
    v_final = ensemble.fuse(v_base, trace.posterior['Delta_V'].values[0])
    
    # 4. 可视化结果
    # 绘制龙卷风图 (Tornado Chart) 逻辑
    print(f"融合后的最终预测分布 (V_final): {v_final}")
    print(f"当前模型确定性权重 (Alpha): {alpha}")

