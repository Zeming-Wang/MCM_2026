import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pylab import mpl
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname="simkai.ttf")

# 正确显示中文文字
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 使用pandas 读取csv文件
df = pd.read_csv('自主呼吸时长.csv')

# 年龄最大和最小值
min_age = df["年龄"].min()
max_age = df["年龄"].max()
print(min_age, max_age)
# 按照年龄分成三组
# 添加年龄分组列（65-70岁和70-75岁）
df["年龄组"] = pd.cut(df["年龄"], bins=[60, 70, 80, 90], labels=["60-70岁", "70-80岁", "80-90岁"])

# 设置绘图风格
sns.set(style="whitegrid")

# 绘制分组箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(
    x="衰弱评分",  # X 轴为衰弱评分
    y="自主呼吸恢复时长",  # Y 轴为恢复时长
    hue="年龄组",  # 按年龄组分组
    data=df,
    palette="Set2",  # 设置颜色
    width=0.6  # 控制箱线图宽度
)

# 添加标题和标签
plt.title("衰弱评分和年龄对自主呼吸恢复时长的影响", fontsize=14, fontproperties=my_font)
plt.xlabel("衰弱评分", fontsize=12, fontproperties=my_font)
plt.ylabel("自主呼吸恢复时长（分钟）", fontsize=12, fontproperties=my_font)
plt.legend(title="年龄组", loc="upper right", prop=my_font)

# 显示图表
plt.savefig('自主呼吸时长箱线图年龄分组.png')
plt.show()



# 添加年龄分组列（65-70岁和70-75岁）
df["衰弱评分组"] = pd.cut(df["衰弱评分"], bins=[0, 1, 3, 4], labels=["60-70岁", "70-80岁", "80-90岁"])

# 设置绘图风格
sns.set(style="whitegrid")

# 绘制分组箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(
    x="年龄",  # X 轴为衰弱评分
    y="自主呼吸恢复时长",  # Y 轴为恢复时长
    hue="衰弱评分组",  # 按衰弱评分组
    data=df,
    palette="Set2",  # 设置颜色
    width=0.6  # 控制箱线图宽度
)

# 添加标题和标签
plt.title("衰弱评分和年龄对自主呼吸恢复时长的影响", fontsize=14, fontproperties=my_font)
plt.xlabel("年龄", fontsize=12, fontproperties=my_font)
plt.ylabel("自主呼吸恢复时长（分钟）", fontsize=12, fontproperties=my_font)
plt.legend(title="衰弱评分组", loc="upper right", prop=my_font)

# 显示图表
plt.savefig('自主呼吸时长箱线图衰弱评分组.png')
plt.show()