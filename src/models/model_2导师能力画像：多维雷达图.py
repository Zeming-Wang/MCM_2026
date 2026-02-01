import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# Load Data
input_path = '/Users/a202507/Desktop/2026/data/processed/Processed_DWTS_Data.csv'
try:
    df_comp = pd.read_csv(input_path)
    # Rename for consistency with the script logic
    if 'ballroom_partner' in df_comp.columns:
        df_comp.rename(columns={'ballroom_partner': 'pro_dancer'}, inplace=True)
except FileNotFoundError:
    print(f"Error: File not found at {input_path}")
    # Create dummy data for testing if file missing
    df_comp = pd.DataFrame({
        'pro_dancer': ['Pro A', 'Pro B', 'Pro A'],
        'growth_slope': [0.5, 0.6, 0.7],
        'celebrity_name': ['Celeb 1', 'Celeb 2', 'Celeb 3']
    })

# Calculate pro_social_spillover (simulating logic from main model)
pro_social_map = {'Mark Ballas': 0.9, 'Sasha Farber': 0.8, 'Witney Carson': 0.85, 'Alan Bersten': 0.7}
df_comp['pro_social_spillover'] = df_comp['pro_dancer'].map(pro_social_map).fillna(0.5)

# Calculate IIS Results (Influence Score)
def calculate_iis(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        slope = row.get('growth_slope', 0)
        social = row.get('pro_social_spillover', 0.5)
        combined_weight = slope * (1 + social)
        G.add_edge(row['pro_dancer'], row['celebrity_name'], weight=combined_weight)
    
    try:
        centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-06)
    except:
        centrality = nx.degree_centrality(G)
        
    iis_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['IIS'])
    iis_df['IIS'] = MinMaxScaler().fit_transform(iis_df[['IIS']])
    # Filter only pros (assuming pros are not in celebrity_name list or strictly by role if available)
    # Here we assume keys in centrality that are in 'pro_dancer' column are pros
    pro_list = df['pro_dancer'].unique()
    return iis_df[iis_df.index.isin(pro_list)]

iis_results = calculate_iis(df_comp)

def plot_pro_radar(iis_scores, df_comp):
    # 1. 数据准备：聚合每个导师的平均表现
    radar_data = df_comp.groupby('pro_dancer').agg({
        'growth_slope': 'mean',
        'pro_social_spillover': 'mean'
    })
    radar_data = radar_data.merge(iis_scores, left_index=True, right_index=True)
    
    # 归一化处理 (0-1)
    radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
    labels = ['Growth Tech', 'Social Spillover', 'Global IIS']
    num_vars = len(labels)

    # 2. 绘图逻辑
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # 闭合多边形

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for pro in radar_norm.index:
        values = radar_norm.loc[pro].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=pro)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title("Pro Dancer Profile: Tech vs. Popularity", y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()

plot_pro_radar(iis_results, df_comp)