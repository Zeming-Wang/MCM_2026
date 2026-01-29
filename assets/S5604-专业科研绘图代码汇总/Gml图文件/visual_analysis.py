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
    # net.show_buttons(filter_=['physics'])  # 启用物理模拟控制
    # 保存为HTML
    net.save_graph(f"analysis1_step{idx}-1.html")
