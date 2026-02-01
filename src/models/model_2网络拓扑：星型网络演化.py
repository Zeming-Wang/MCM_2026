import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load Data
input_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Data.csv'
try:
    df_comp = pd.read_csv(input_path)
    if 'ballroom_partner' in df_comp.columns:
        df_comp.rename(columns={'ballroom_partner': 'pro_dancer'}, inplace=True)
except FileNotFoundError:
    print(f"Error: File not found at {input_path}")
    df_comp = pd.DataFrame({
        'pro_dancer': ['Pro A', 'Pro B'],
        'celebrity': ['Celeb 1', 'Celeb 2'],
        'growth_slope': [0.5, 0.6]
    })
    # ensure celebrity column exists if it was named something else in loaded csv
    if 'celebrity_name' in df_comp.columns:
         df_comp.rename(columns={'celebrity_name': 'celebrity'}, inplace=True)

# Ensure celebrity column is consistent (CSV has 'celebrity_name')
if 'celebrity_name' in df_comp.columns:
    df_comp['celebrity'] = df_comp['celebrity_name']

# Calculate IIS for coloring/sizing
def calculate_iis(df):
    G = nx.DiGraph()
    pro_social_map = {'Mark Ballas': 0.9, 'Sasha Farber': 0.8, 'Witney Carson': 0.85, 'Alan Bersten': 0.7}
    df['pro_social_weight'] = df['pro_dancer'].map(pro_social_map).fillna(0.5)
    
    for _, row in df.iterrows():
        slope = row.get('growth_slope', 0)
        social = row.get('pro_social_weight', 0.5)
        combined_weight = slope * (1 + social)
        G.add_edge(row['pro_dancer'], row['celebrity'], weight=combined_weight)
    
    try:
        centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-06)
    except:
        centrality = nx.degree_centrality(G)
        
    iis_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['IIS'])
    iis_df['IIS'] = MinMaxScaler().fit_transform(iis_df[['IIS']])
    pro_list = df['pro_dancer'].unique()
    return iis_df[iis_df.index.isin(pro_list)]

iis_results = calculate_iis(df_comp)

def generate_network_viz(df_comp, iis_scores):
    G = nx.Graph()
    
    # 添加导师节点 (核心)
    for pro, score in iis_scores.iterrows():
        G.add_node(pro, size=score['IIS']*1000, type='pro', color='gold')
    
    # 添加选手节点并连线
    for _, row in df_comp.iterrows():
        G.add_node(row['celebrity'], size=200, type='celeb', color='skyblue')
        # 边权：进步速度越快，连线越粗
        G.add_edge(row['pro_dancer'], row['celebrity'], weight=row['growth_slope'])

    # 1. Python 内部预览绘图
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5) # 使用弹簧布局
    
    # 提取属性用于绘图
    node_sizes = [G.nodes[n]['size'] for n in G.nodes]
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    edge_weights = [G.edges[e]['weight'] * 5 for e in G.edges]

    nx.draw(G, pos, with_labels=True, node_size=node_sizes, 
            node_color=node_colors, width=edge_weights, edge_color='gray', alpha=0.8)
    plt.title("Star-Network: Instructor Influence & Growth Linkage")
    plt.show()

    # 2. 导出为 Gephi 格式
    nx.write_gexf(G, "DWTS_Influence_Network.gexf")
    print("Success: 'DWTS_Influence_Network.gexf' has been generated for Gephi.")

generate_network_viz(df_comp, iis_results)