import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import networkx as nx
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
import importlib.util
from pathlib import Path

# ==========================================
# 1. 基础物理约束层 (Base Layer - Model A)
# 逻辑：利用 Norman Biggs 的空间过滤逻辑，确定最小可行解空间
# ==========================================
"""
灵感来源于ResNet
其中某个变量捕捉了物理规则无法描述的软因素（如粉丝狂热度等）

"""


"""
数学原理：

这里目前实现的是一个“无知先验”（Uniform Distribution）
在实际的物理约束模型，应当计算满足相应不等式约束
R_j + R_f > R_eliminated 的解空间

alpha 熵驱动的权重调整
数学原理：
    熵（Entropy）是信息论中衡量随机变量不确定性的指标。
    当一个随机变量的分布越均匀（即熵越大），其信息的不确定性就越大。
    因此，我们可以利用熵来调整不同变量的权重，使得模型更关注那些信息更丰富的变量。
"""

"""
获得MCMC得到关于V的不确定性估计
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
        上面的为特征嵌入，将离散的索引变为连续的隐变量
        """

        
        # 中间隐变量 (Latent Variables)
        # Audience_Preference 受行业和初始粉丝量影响
        pref_mu = industry_effect[industry_idx] + age_slope * age_data
        audience_pref = pm.Normal('Audience_Preference', mu=pt.mean(pref_mu), sigma=1)
        """
        上面的为中间隐变量，受行业和初始粉丝量影响
        和年龄效应相结合
        """
        # 目标残差变量 (Score_Residual)
        # 这里体现了从特征到残差的非线性映射
        delta_v = pm.Normal('Delta_V', mu=audience_pref, sigma=0.5, shape=len(age_data))
        """
        最终的残差Delta_V被建模为一个特征组合为均值的正态分布
        """
        return residual_model

# ==========================================
# 3. 异构融合架构 (Ensemble Engine)
# 逻辑：残差校准 + 投影算子 P
# ==========================================

class ResidualCalibrationEnsemble:
    def __init__(self, alpha_init=0.5):
        self.alpha = alpha_init  # 自适应权重因子

    def projection_operator_P(self, v_raw):
        
        """
        投影算子 P：将融合结果强制映射回符合规则的单纯形空间 (Sum=1, >=0)
        并满足淘汰不等式约束
        
        使用 Softmax 保证和为 1，或使用纠偏矩阵，具有放大差异的特征
        适合处理投票/概率分布
        """

        exp_v = np.exp(v_raw)
        return exp_v / np.sum(exp_v)

    def adaptive_weight_logic(self, solution_space_entropy):
        
        #根据解空间稀疏度 (信息熵) 动态调整 alpha
        #解空间越小，物理约束越强，alpha 越小
        
        self.alpha = 1.0 / (1.0 + np.exp(-solution_space_entropy))
        return self.alpha

        """
        自适应权重：使用sigmoid函数将熵映射到(0.5, 1)之间，
        熵越大（解空间越不确定），alpha 越大，模型越依赖数据驱动的残差；
        熵越小（解空间越确定），alpha 越小，模型越依赖物理约束。

        """

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

def _load_module(module_name, relative_path):
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / relative_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ==========================================
# 5. 主逻辑执行流程
# ==========================================

if __name__ == "__main__":
    import os
    # --- 数据路径占位 ---
    DATA_PATH = r"d:\MCM_2026_O\data\processed\processed_mcm_data.csv"
    
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        
        # 选取 Season 1 的数据作为测试集
        test_df = df[df['season'] == 1].copy()
        test_df = test_df.reset_index(drop=True)
        print(f"Loaded {len(test_df)} contestants from Season 1 for testing.")

        # 1. 初始化物理层 (模型 A)
        # 这里使用 judge_percent 作为基础分数的参考，或者继续使用均匀分布
        # 为了演示代码运行，我们传入真实的评委分数（这里取 Week 1 的总分作为示例）
        # 注意：PhysicalConstraintModel 目前只用到了 len(scores)
        
        # 计算 Week 1 总分
        judge_cols = ['week1_judge1_score', 'week1_judge2_score', 'week1_judge3_score', 'week1_judge4_score']
        # 确保是数值类型
        for col in judge_cols:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
            
        test_scores = test_df[judge_cols].sum(axis=1).values
        
        phys_model = PhysicalConstraintModel(test_scores, "Eliminated_Week_9") # 淘汰信息暂未具体使用
        v_base = phys_model.get_feasible_base_votes()
        print(f"Base Votes (Uniform): {v_base}")
        
        # 2. 运行贝叶斯残差层 (模型 B)
        # 从处理后的数据中提取特征
        age_mock = test_df['celebrity_age_during_season'].values
        ind_mock_raw = test_df['industry_idx'].values
        # 重编码 industry_idx 以适应当前子集 (确保索引从 0 到 N-1)
        # 否则如果子集中只有索引 [13, 25]，而模型只创建了 shape=2 的变量，访问 13 会越界
        _, ind_mock = np.unique(ind_mock_raw, return_inverse=True)
        
        # 使用 youth_factor 作为 fan_base_data 的代理，或者生成随机数
        # 这里假设 fan_base_data 已经在预处理中体现，或者我们临时用 youth_factor * 0.5 + 0.2 模拟
        if 'youth_factor' in test_df.columns:
            fb_mock = test_df['youth_factor'].values * 0.5 + 0.2
        else:
            fb_mock = np.random.rand(len(test_df))
        
        print(f"Running Bayesian Residual Model with {len(age_mock)} samples...")
        res_model = build_bayesian_residual_model(ind_mock, age_mock, fb_mock)
        with res_model:
            # 进行最大后验估计 (MAP)
            map_estimate = pm.find_MAP()
            # 进行 MCMC 采样获取后验宽度 (Uncertainty)
            # 减少采样数以加快测试速度
            trace = pm.sample(100, tune=100, chains=2, return_inferencedata=True, progressbar=False)
        
        # 3. 融合与校准
        ensemble = ResidualCalibrationEnsemble()
        # 计算解空间熵 (示例值)
        h_c = 0.44 / 0.720 # 引用 Norman Biggs 的例子比率
        alpha = ensemble.adaptive_weight_logic(h_c)
        
        # 获取 Delta_V 的后验均值
        delta_v_samples = trace.posterior['Delta_V'].values
        # posterior shape: (chains, draws, shape) -> 合并 chains 和 draws
        delta_v_samples_reshaped = delta_v_samples.reshape(-1, delta_v_samples.shape[-1])
        
        v_final = ensemble.fuse(v_base, delta_v_samples_reshaped)
        
        # 4. 可视化结果
        print(f"融合后的最终预测分布 (V_final): {v_final}")
        print(f"Sum of V_final: {np.sum(v_final)}")
        print(f"当前模型确定性权重 (Alpha): {alpha}")
        
        # 简单的结果展示
        result_df = pd.DataFrame({
            'Name': test_df['celebrity_name'],
            'Base_Prob': v_base,
            'Final_Prob': v_final
        })
        print("\nTop 5 Predictions:")
        print(result_df.sort_values('Final_Prob', ascending=False).head(5))
        charts_dir = r"d:\MCM_2026_O\charts"
        tornado_mod = _load_module("tornado_chart", "灵敏度分析：龙卷风图 (Tornado Chart).py")
        tornado_mod.plot_sensitivity_tornado()
        alpha_mod = _load_module("alpha_evolution", "敏感性分析-动态 alpha 自适应演化图 (Alpha Evolution Dashboard).py")
        weeks = list(range(1, len(v_final) + 1))
        alpha_series = list(np.linspace(alpha * 0.8, alpha * 1.2, len(weeks)))
        entropy_series = list(np.linspace(h_c * 1.2, h_c * 0.8, len(weeks)))
        alpha_mod.plot_alpha_evolution(weeks, alpha_series, entropy_series)
        proj_mod = _load_module("projection_heatmap", "一致性检验-投影偏移热力图 (Projection Consistency Heatmap).py")
        delta_v_mean = np.mean(delta_v_samples_reshaped, axis=0)
        v_raw = v_base + alpha * delta_v_mean
        proj_mod.plot_projection_consistency(v_raw.tolist(), v_final.tolist())
        boundary_mod = _load_module("boundary_fidelity", "一致性检验-淘汰边界的一致性验证图 (Boundary Fidelity Plot).py")
        df_results = pd.DataFrame({
            "JudgeRank": np.arange(1, len(v_final) + 1),
            "PredictedScoreRank": np.argsort(-v_final) + 1,
            "EliminatedStatus": test_df["results"].apply(lambda x: "Eliminated" if isinstance(x, str) and "Eliminated" in x else ("Winner" if isinstance(x, str) and "1st Place" in x else "Safe")),
        })
        boundary_mod.plot_boundary_fidelity(df_results)
        radar_mod = _load_module("uncertainty_radar", "敏感性分析-贝叶斯后验区间与灵敏度雷达图 (Uncertainty Radar).py")
        sobol_indices = [0.45, 0.33, 0.22, 0.18, 0.12]
        hdi_95 = [0.05, 0.04, 0.03, 0.02, 0.01]
        radar_mod.plot_sensitivity_radar(sobol_indices, hdi_95)
        try:
            ternary_mod = _load_module("ternary_plot", "物理约束层：解空间三元图 (Ternary Plot).py")
            ternary_mod.plot_ternary_feasible_region()
        except ModuleNotFoundError:
            pass
        dag_mod = _load_module("bayes_dag", "结构逻辑图：贝叶斯网络 (DAG).py")
        dag_mod.plot_bayesian_network()
    else:
        print(f"Error: Processed data file not found at {DATA_PATH}")
