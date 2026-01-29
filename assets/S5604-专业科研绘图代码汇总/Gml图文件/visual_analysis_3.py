import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain  # 需要安装 python-louvain 包

# 读取所有时间步文件
time_steps = []
for i in range(6):
    file_path = f"primaryschool/primaryschool_s_{i}.gml"
    G = nx.read_gml(file_path, label="id")  # 确保节点使用 id 字段作为标识
    time_steps.append(G)


# 为每个时间步计算社区
communities_list = []
for G in time_steps:
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    communities_list.append(communities)

# 绘制带社区颜色的网络图
for idx, (G, communities) in enumerate(zip(time_steps, communities_list)):
    plt.figure(figsize=(10, 8))

    # 创建颜色映射
    colors = []
    for node in G.nodes():
        for comm_id, members in communities.items():
            if node in members:
                colors.append(comm_id)
                break

    pos = nx.spring_layout(G, seed=42)  # 保持各时间步布局一致
    nx.draw(G, pos, node_color=colors, cmap=plt.cm.tab20,
            with_labels=False, node_size=50)
    plt.title(f"Time Step {idx} - Louvain Communities")
    plt.savefig(f"analysis1_step{idx}-2.png", dpi=300)
    plt.close()


# 计算每个节点的 k-core 值
kcore_results = []
for G in time_steps:
    kcores = nx.k_core(G)
    kcore_values = nx.core_number(G)
    kcore_results.append(kcore_values)

# 将结果写入 GEXF 格式（Gephi可识别）
for idx, (G, kcore) in enumerate(zip(time_steps, kcore_results)):
    nx.set_node_attributes(G, kcore, "kcore")
    nx.write_gexf(G, f"analysis2_step{idx}.gexf")

import imageio

# 创建动画GIF
images = []
for i in range(6):
    images.append(imageio.imread(f"analysis1_step{i}-2.png"))
imageio.mimsave('analysis1.gif', images, fps=1)
