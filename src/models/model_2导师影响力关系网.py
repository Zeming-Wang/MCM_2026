import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler

# --- 1. 学术风格全局配置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 美赛首选学术字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300  # 确保高清输出

# 定义输出目录
output_dir = '/Users/a202507/Desktop/2026/data/processed/plots'
os.makedirs(output_dir, exist_ok=True)

def plot_iis_network_pro(master_model):
    """
    O奖标准：导师-明星影响力网络
    优化点：增加背景浅色标注（水印式文本）、层次化权重、学术色系
    """
    # 提取数据
    df = master_model.analysis_df
    iis_scores = master_model.iis_results['IIS'].to_dict()
    
    G = nx.Graph()
    
    # --- 2. 边权重与节点属性构建 ---
    for _, row in df.iterrows():
        pro = row['ballroom_partner']
        celeb = row['celebrity_name']
        # 使用增长斜率作为边权重，影响布局的紧密度
        weight = max(0.1, row.get('growth_slope', 0.5))
        G.add_edge(pro, celeb, weight=weight)
        
    # --- 3. 节点与色彩逻辑 (使用学术冷色调) ---
    node_sizes = []
    node_colors = []
    labels = {}
    
    # 颜色常数：深藏蓝代表专业导师，浅灰蓝代表明星
    COLOR_PRO = "#FFF2D0" 
    COLOR_CELEB = "#A9D9ED"
    
    for node in G.nodes():
        if node in iis_scores: # 导师
            # 节点大小由 IIS 决定，范围在 500-3000
            score = iis_scores[node]
            node_sizes.append(600 + score * 2500)
            node_colors.append(COLOR_PRO)
            # 仅标注 IIS 较高的导师，避免视觉拥挤
            if score > np.percentile(list(iis_scores.values()), 50):
                labels[node] = node
        else: # 明星
            node_sizes.append(200)
            node_colors.append(COLOR_CELEB)

    # --- 4. 高级布局算法 ---
    plt.figure(figsize=(13, 11))
    # 使用 Kamada-Kawai 布局使结构更平衡，或使用带权重的 spring_layout
    pos = nx.spring_layout(G, k=0.25, iterations=60, seed=42, weight='weight')
    
    # --- 【新增：高级感背景文字（Shadow Text）】 ---
    bg_font_style = {'fontsize': 45, 'color': '#E0E0E0', 'alpha': 0.3, 
                     'fontweight': 'bold', 'fontstyle': 'italic', 'ha': 'center'}
    
    plt.text(0.5, 0.9, "TECHNIQUE DRIVEN", transform=plt.gca().transAxes, **bg_font_style)
    plt.text(0.1, 0.1, "SOCIAL FLOW", transform=plt.gca().transAxes, **bg_font_style)
    plt.text(0.85, 0.15, "SYNERGY", transform=plt.gca().transAxes, **bg_font_style)

    # --- 5. 分层绘制 ---
    # 绘制边：透明度随权重变化，体现“影响力流”
    nx.draw_networkx_edges(G, pos, width=1.2, alpha=0.15, edge_color="#AEDCF1")
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                           node_color=node_colors, 
                           edgecolors='#B0BEC5', linewidths=0.5, alpha=0.9)
    
    # 绘制标签：仅显示核心导师
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, 
                            font_weight='bold', font_family='serif', font_color="#425B67") # 颜色加深一点以提高对比度

    # --- 6. 增加自解释图例 (O奖必杀技) ---
    pro_patch = mpatches.Patch(color=COLOR_PRO, label='Professional Mentor (Size ∝ IIS)')
    celeb_patch = mpatches.Patch(color=COLOR_CELEB, label='Celebrity Contestant')
    plt.legend(handles=[pro_patch, celeb_patch], loc='lower right', 
               frameon=False, fontsize=11)

    # --- 7. 细节修饰 ---
    plt.title("Figure 1. Topological Influence Network with Latent Dimension Annotations\n"
              "Node scale derived from Integrated Influence Score (IIS)", 
              fontsize=18, pad=30, fontweight='bold', fontfamily='serif')
    
    plt.axis('off')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'Academic_Mentor_Network_Final.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"O奖标准最终版网络图已导出至: {output_file}")
    plt.close()

# --- Helper Class for Data Processing ---
class DWTSMasterModel:
    def __init__(self, df):
        self.df = df
        self.iis_results = None
        self.analysis_df = df  # Use raw df as analysis df for now

    def feature_engineering(self):
        print("Executing Feature Engineering...")
        # D. 导师社交权重
        pro_social_map = {'Mark Ballas': 0.9, 'Sasha Farber': 0.8, 'Witney Carson': 0.85, 'Alan Bersten': 0.7}
        self.df['pro_social_weight'] = self.df['ballroom_partner'].map(pro_social_map).fillna(0.5)

    def run_network_layer(self):
        print("Running Network Layer (IIS Calculation)...")
        G = nx.DiGraph()
        
        for _, row in self.df.iterrows():
            slope = row.get('growth_slope', 0)
            social_multiplier = row['pro_social_weight']
            combined_weight = slope * (1 + social_multiplier)
            G.add_edge(row['ballroom_partner'], row['celebrity_name'], weight=combined_weight)
        
        try:
            centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-06)
            iis_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['IIS'])
            iis_df['IIS'] = MinMaxScaler().fit_transform(iis_df[['IIS']])
            # Filter to keep only mentors (those not in celebrity list)
            celebrities = set(self.df['celebrity_name'].unique())
            self.iis_results = iis_df[~iis_df.index.isin(celebrities)]
        except Exception as e:
            print(f"Network error (eigenvector): {e}")
            centrality = nx.degree_centrality(G)
            iis_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['IIS'])
            self.iis_results = iis_df[~iis_df.index.isin(self.df['celebrity_name'])]
        
        return self.iis_results

if __name__ == "__main__":
    input_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Data.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        exit(1)
        
    df = pd.read_csv(input_path)
    model = DWTSMasterModel(df)
    model.feature_engineering()
    model.run_network_layer()
    
    plot_iis_network_pro(model)
