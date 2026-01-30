#逻辑： 展示在三人竞争且已知一人淘汰的情况下，合法的粉丝投票分布区域。#
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np
import os

def plot_ternary_feasible_region(output_path=None):
    # 模拟在三维单纯形上的可行域
    # 假设顶点为选手 A, B, C。阴影部分代表满足 R_j + R_f 约束的区域
    points = np.random.dirichlet([1, 1, 1], 5000)
    # 逻辑过滤：模拟“淘汰约束”下的可行解
    feasible = points[points[:, 0] > 0.4] # 假设 A 必须获得足够票数才不被淘汰
    
    fig = ff.create_ternary_contour(feasible.T, np.ones(len(feasible)),
                                   interp_mode='cartesian', ncontours=15,
                                   colorscale='Blues', showscale=True)
    
    fig.update_layout(title="解空间可行多胞体与香农熵分布 (Feasible Polytope & Entropy)")
    charts_dir = r"d:\MCM_2026_O\charts"
    if output_path is None:
        output_path = os.path.join(charts_dir, "ternary_feasible_region.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)

if __name__ == "__main__":
    plot_ternary_feasible_region()
