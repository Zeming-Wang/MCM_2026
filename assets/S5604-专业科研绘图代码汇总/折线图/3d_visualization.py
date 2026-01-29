import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'  # 全局加粗
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold'  # 标题加粗

# 读取CSV文件
df = pd.read_csv('samples_data_Bearing1_1.csv')  # 替换为你的CSV文件路径

# 提取s1-s50列的数据
s_columns = [f's{i}' for i in range(1, 51)]
data = df[s_columns].values

# 创建3D图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([2, 3, 0.7])  # [x, y, z] 比例，调整y值使y轴更长
ax.grid(False)
# 获取行索引和列索引
rows = np.arange(data.shape[0])  # Y轴（行索引）
print(data.shape) # (2801, 50)
cols = np.arange(data.shape[1])  # X轴（列索引）

# 为了绘制的图形不挡住其他线条
# 按照s1-s50计算总的和，然后按照和从小到大排序
sums = data.sum(axis=0)
print("sums:", sums, sums.shape)
indices = sums.argsort()
print(indices)
indices_cols = indices[::-1]
print(indices_cols)

# 为每条曲线设置不同的X轴位置
x_positions = np.linspace(0, 50, len(cols))  # 在X轴上均匀分布
print(x_positions)
# 绘制每条曲线
for i, col in enumerate(cols):
    # 每条曲线在X轴上的位置相同，Y轴是行索引，Z轴是数据值
    ax.plot(xs=np.full_like(rows, x_positions[i]),  # X坐标相同
            ys=rows,  # Y坐标是行索引
            zs=data[:, indices_cols[i]],  # Z坐标是数据值
            label=f's{i+1}')

# 设置坐标轴标签
# ax.set_xlabel('Sensor Columns (s1-s50)')
ax.set_ylabel('Row Index', labelpad=15)
ax.set_zlabel('Sensor Values')
ax.set_ylim(0, len(rows)-1)

# 设置平面颜色
ax.xaxis.set_pane_color((0.8, 0.9, 1.0, 0.5))  # YZ平面（浅蓝色，50%透明度）
ax.yaxis.set_pane_color((0.7, 0.9, 1.0, 0.5))  # XZ平面
ax.zaxis.set_pane_color((0.9, 0.8, 0.7, 0.5))  # XY平面

ax.xaxis._axinfo["grid"].update({"visible": True})  # 单独取消X轴网格
ax.yaxis._axinfo["grid"].update({"visible": False})  # 单独取消Y轴网格
ax.zaxis._axinfo["grid"].update({"visible": False})  # 单独取消Z轴网格

# 添加图例（右侧分两列）
ax.legend(bbox_to_anchor=(0.5, 0.754),
          title='Curves',
          loc='lower center',
          fontsize=9,
          ncol=10,
          edgecolor='black',
          frameon=True,
          facecolor='#ecfbc7',    # 背景色（白色）
          )
# 设置X轴刻度为列名
# ax.set_xticks(x_positions[::5])  # 每隔5个显示一个标签
# ax.set_xticklabels([f's{i+1}' for i in range(0, 50, 5)])

# 调整视角以便更好地查看
ax.view_init(elev=20, azim=-45)

plt.title('3D Visualization of s1-s50 Columns')
plt.tight_layout()
plt.savefig('3d_visualization.png', dpi=400)
plt.show()