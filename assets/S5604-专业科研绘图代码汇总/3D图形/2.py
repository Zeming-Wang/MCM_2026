import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize  # 导入Normalize用于颜色标准化

# 设置字体路径
font_path = r'../simkai.ttf'

# 加载字体并设置
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
x = np.linspace(-4, 4, 5)
y = np.linspace(-4, 4, 5)
X, Y = np.meshgrid(x, y)

# 生成凹凸表面
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        a = np.random.random()
        print(a)
        Z[i, j] = a * 1

# 创建图形
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 创建颜色标准化对象
# norm = Normalize(vmin=-0.2, vmax=0.2)

# 绘制表面图，设置颜色范围
surf = ax.plot_surface(
    X, Y, Z,
    cmap=cm.viridis,
    edgecolor='k',
    linewidth=0.3,
    rstride=3,
    cstride=3,
    # norm=norm  # 应用颜色标准化
)

# 添加颜色条
cbar = fig.colorbar(surf, shrink=0.6, aspect=8)
cbar.set_label('高度值', rotation=270, labelpad=15)

# 设置z轴范围
ax.set_zlim(-1, 2)

# 设置视角
ax.view_init(elev=40, azim=30)

# 添加标签
ax.set_xlabel('X轴', labelpad=10)
ax.set_ylabel('Y轴', labelpad=10)
ax.set_zlabel('Z轴', labelpad=10)
ax.set_title('简化版三维表面', pad=15)

plt.tight_layout()
plt.show()