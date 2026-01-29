import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 创建数据
x = np.linspace(-4, 4, 13)
y = np.linspace(-4, 4, 13)
X, Y = np.meshgrid(x, y)

# 生成凹凸表面
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        a = np.random.random() - 0.5
        print(a)
        Z[i, j] = a * 1.5

# 创建图形
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制表面图
surf = ax.plot_surface(
    X, Y, Z,
    cmap=cm.viridis,
    edgecolor='k',  # 添加黑色网格线
    linewidth=0.3,
    rstride=3,  # 增大行跨度
    cstride=3  # 增大列跨度
)

# 创建5×5的散点数据（z=1高度）
scatter_x = np.linspace(-4, 4, 5)  # -4到4均匀分布5个点
scatter_y = np.linspace(-4, 4, 5)
scatter_X, scatter_Y = np.meshgrid(scatter_x, scatter_y)
scatter_Z = np.ones_like(scatter_X)  # 所有点z坐标为1

# 绘制z=1的散点（蓝色'x'标记）
scatter = ax.scatter(
    scatter_X, scatter_Y, scatter_Z,
    marker='x',       # 使用x标记
    color='blue',      # 蓝色标记
    s=40,           # 标记大小
    # label='散点'
)

# 添加颜色条
cbar = fig.colorbar(surf, shrink=0.6, aspect=8)
cbar.set_label('height', rotation=270, labelpad=15)

# 设置视角
ax.view_init(elev=30, azim=30)

# 添加标签
ax.set_xlabel('X Surface Position', labelpad=10)
ax.set_ylabel('Y Surface Position', labelpad=10)
ax.set_zlabel('Z Energy intensity of the adsorption site', labelpad=10)
ax.set_title('Non-uniform adsorption surface', pad=15)
ax.set_zlim(-1, 2)
plt.tight_layout()
plt.savefig('3.png', dpi=400, bbox_inches='tight')
plt.show()