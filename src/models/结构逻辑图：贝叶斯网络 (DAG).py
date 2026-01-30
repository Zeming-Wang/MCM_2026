#逻辑： 用于向评审展示特征如何流向残差校准项 V。#
import matplotlib.pyplot as plt
import networkx as nx
import os

def plot_bayesian_network(output_path=None):
    plt.figure(figsize=(10, 6), dpi=120)
    G = nx.DiGraph()
    
    # 定义节点
    nodes = {
        'Industry': '明星行业类别\n(Industry)',
        'Age': '选手年龄\n(Age)',
        'FanBase': '初始粉丝基数\n(Initial Fan Base)',
        'Pref': '受众偏好程度\n(Latent Preference)',
        'Vol': '投票波动率\n(Volatility)',
        'DeltaV': '投票残差\n(Residual ΔV)'
    }
    
    # 添加边 (体现 V-Structure)
    edges = [('Industry', 'Pref'), ('Age', 'Pref'), ('FanBase', 'Pref'),
             ('Pref', 'DeltaV'), ('Vol', 'DeltaV')]
    
    pos = {
        'Industry': (0, 2), 'Age': (0, 1), 'FanBase': (0, 0),
        'Pref': (2, 1), 'Vol': (2, 0),
        'DeltaV': (4, 0.5)
    }
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#F0F4F8', edgecolors="#1A86F2")
    nx.draw_networkx_labels(G, pos, labels=nodes, font_size=9, font_family='SimHei')
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=20, edge_color="#2DC0CB", width=1.5)
    
    # 标注数学符号
    plt.text(1, 1.7, r'$\mu_{pref}$', fontsize=12, color='blue')
    plt.text(3, 1, r'$\sigma_{residual}$', fontsize=12, color='red')
    
    plt.title("贝叶斯网络残差生成机制 (Bayesian Residual Architecture)", fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    charts_dir = r"d:\MCM_2026_O\charts"
    if output_path is None:
        output_path = os.path.join(charts_dir, "bayesian_network_dag.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_bayesian_network()
