import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage

import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain  # Louvain算法库
from pyvis.network import Network
import imageio  # 用于生成GIF

# 读取所有时间步数据
time_steps = []
for i in range(6):
    G = nx.read_gml(f"primaryschool/primaryschool_s_{i}.gml")
    time_steps.append(G)

# 社区检测与可视化
for idx, G in enumerate(time_steps):
    # Louvain算法检测社区
    partition = community_louvain.best_partition(G)

    # 创建PyVis网络
    net = Network(notebook=True, height="600px", width="800px")
    net.from_nx(G)

    # 根据社区着色
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD"]
    for node in net.nodes:
        node["color"] = colors[partition[node["id"]] % len(colors)]
        node["size"] = G.degree(node["id"]) * 2  # 节点大小与度数相关
    net.show_buttons(filter_=['physics'])  # 启用物理模拟控制
    # 保存为HTML
    net.save_graph(f"analysis1_step{idx}-1.html")


# 构建邻接矩阵序列
adj_matrices = [nx.to_numpy_array(G) for G in time_steps]

# 层次聚类分析
for idx, mat in enumerate(adj_matrices):
    plt.figure(figsize=(10, 8))

    # 创建聚类链接矩阵
    Z = linkage(mat, method='ward')

    # 绘制聚类热力图
    clustergrid = sns.clustermap(
        pd.DataFrame(mat),
        row_linkage=Z,
        col_linkage=Z,
        cmap="YlGnBu",
        figsize=(12, 10),
        metric='cosine',  # 改用余弦相似度
        dendrogram_ratio=0.2,  # 调整树状图比例
        cbar_pos=(0.02, 0.8, 0.03, 0.18)  # 重定位颜色条
    )
    clustergrid.ax_heatmap.set_title(f"Time Step {idx} - Hierarchical Clustering")
    plt.savefig(f"analysis2_step{idx}.png", dpi=150)
    plt.close()


analysis1_images = []
for i in range(6):
    img = imageio.v2.imread(f"analysis2_step{i}.png")
    analysis1_images.append(img)
imageio.mimsave('analysis2_animation.gif', analysis1_images, fps=1.5)