# 导入相关库
import numpy as np
import matplotlib.pyplot as plt
import funcy
import warnings
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# 设置字体路径
font_path = r'../simkai.ttf'

# 加载字体并设置
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 导入示例数据
# 数据来源网络问卷
majors = ['营销', '金融', '心理学', '新闻学', '电视', '日语', '汉语言', '物理', '机械', '文学', '生物', '计算机',
          '化学', '数学', '设计', '英语', '会计', '教育', '工程', '医学']
datas = [18.7, 21.6, 22.9, 23.5, 24.8, 26.2, 27.1, 27.1, 29.6, 31.3, 32.6, 33.5, 36.3, 40.8, 54.3, 55.8, 70.1, 70.3,
         70.7, 100]

# 绘制图布，设置极坐标系
fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
ax1 = fig.add_axes([0.22, 0.2, 0.6, 0.6], polar=True)

theta_group = np.linspace(0, 1, 21) * np.pi
# 绘制上半扇形区域
for idx, group in enumerate(funcy.pairwise(theta_group)):
    ax1.fill_between(group, 1.4, 4, facecolor='#fafbff' if idx % 2 == 0 else '#e3effd')

fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
ax1 = fig.add_axes([0.22, 0.2, 0.6, 0.6], polar=True)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
[ax.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]

theta_group = np.linspace(0, 1, 21) * np.pi
# 绘制上半扇形区域
for idx, group in enumerate(funcy.pairwise(theta_group)):
    ax1.fill_between(group, 1.4, 4, facecolor='#fafbff' if idx % 2 == 0 else '#e3effd')

# 绘制虚线
for idx, group in enumerate(funcy.pairwise(theta_group)):
    theta = (group[0] + group[1]) / 2
    ax1.plot([theta, theta], [1.4, 3.5], linestyle='dashed', color='#9fa0a0', linewidth=0.25)

fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
ax1 = fig.add_axes([0.22, 0.2, 0.6, 0.6], polar=True)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
[ax.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]

theta_group = np.linspace(0, 1, 21) * np.pi
# 绘制扇形区域
for idx, group in enumerate(funcy.pairwise(theta_group)):
    ax1.fill_between(group, 1.4, 4, facecolor='#fafbff' if idx % 2 == 0 else '#e3effd')

# 绘制虚线
for idx, group in enumerate(funcy.pairwise(theta_group)):
    theta = (group[0] + group[1]) / 2
    ax1.plot([theta, theta], [1.4, 3.5], linestyle='dashed', color='#9fa0a0', linewidth=0.25)

# 绘制中央虚线
for idx, group in enumerate(funcy.pairwise(theta_group)):
    theta = (group[0] + group[1]) / 2
    ax1.plot([theta, theta], [1.4, 3.5], linestyle='dashed', color='#9fa0a0', linewidth=0.25)
    # 柱状图
    ax1.bar(theta, 0.025 * datas[idx], width=[np.pi / 21], bottom=1.4,
            facecolor='#6785f2' if idx % 2 == 0 else '#7171fe', edgecolor='white', linewidth=0.1, alpha=0.95, zorder=9
            )

fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
ax1 = fig.add_axes([0.22, 0.2, 0.6, 0.6], polar=True)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
[ax.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]

theta_group = np.linspace(0, 1, 21) * np.pi
# 绘制扇形区域
for idx, group in enumerate(funcy.pairwise(theta_group)):
    ax1.fill_between(group, 1.4, 4, facecolor='#fafbff' if idx % 2 == 0 else '#e3effd')

# 绘制中央虚线
for idx, group in enumerate(funcy.pairwise(theta_group)):
    theta = (group[0] + group[1]) / 2
    ax1.plot([theta, theta], [1.4, 3.5], linestyle='dashed', color='#9fa0a0', linewidth=0.25)
    # 柱状图
    ax1.bar(theta, 0.025 * datas[idx], width=[np.pi / 21], bottom=1.4,
            facecolor='#6785f2' if idx % 2 == 0 else '#7171fe', edgecolor='white', linewidth=0.1, alpha=0.95, zorder=9
            )

# 绘制扇形区域白色边界
for theta in theta_group:
    ax1.plot([theta, theta], [1, 4], color='w', linewidth=0.5)

fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
ax1 = fig.add_axes([0.22, 0.2, 0.6, 0.6], polar=True)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
[ax.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]

theta_group = np.linspace(0, 1, 21) * np.pi
# 绘制扇形区域
for idx, group in enumerate(funcy.pairwise(theta_group)):
    ax1.fill_between(group, 1.4, 4, facecolor='#fafbff' if idx % 2 == 0 else '#e3effd')

# 绘制中央虚线
for idx, group in enumerate(funcy.pairwise(theta_group)):
    theta = (group[0] + group[1]) / 2
    ax1.plot([theta, theta], [1.4, 3.5], linestyle='dashed', color='#9fa0a0', linewidth=0.25)
    # 柱状图
    ax1.bar(theta, 0.025 * datas[idx], width=[np.pi / 21], bottom=1.4,
            facecolor='#6785f2' if idx % 2 == 0 else '#7171fe', edgecolor='white', linewidth=0.1, alpha=0.95, zorder=9
            )

# 绘制扇形区域白色边界
for theta in theta_group:
    ax1.plot([theta, theta], [1, 4], color='w', linewidth=0.5)

# 文本
for idx, group in enumerate(funcy.pairwise(theta_group)):  # 文字角度
    angle = ((group[0] + group[1]) * 0.5 / np.pi) * 180 - (0 if idx < (len(majors) / 2) else 180)
    # 专业
    ax1.annotate(majors[idx], xy=[(group[0] + group[1]) / 2, 4.4], va='center', ha='center', zorder=10, rotation=angle,
                 fontsize=10)
    # 指数
    ax1.annotate(datas[idx],
                 xy=[(group[0] + group[1]) / 2, 3.79],
                 va='center', ha='center', zorder=10,
                 rotation=angle,
                 fontsize=10
                 )
plt.show()