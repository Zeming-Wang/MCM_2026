import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建DataFrame
df = pd.read_csv('数据.csv')
delete_na_columns = ['最近地铁站', '直线距离', '通行距离', '步行时间（分钟）', '骑行时间（分钟）', '驾车（分钟）']
df = df.dropna(subset=delete_na_columns)
# 创建掩码，标记所有指定列都不为0的行
mask = (df[delete_na_columns] != 0).all(axis=1)
# 应用过滤
df = df[mask]

std_distance = 800
std_time = 10
# 计算与国际标准(800米)的差距
df['超标准距离'] = df['通行距离'] - std_distance
df['超标准时间'] = df['步行时间（分钟）'] - std_time  # 国际标准12分钟

# 基本统计
print(df[['通行距离', '步行时间（分钟）', '超标准距离', '超标准时间']].describe())

plt.figure(figsize=(12, 6), dpi=400)

# 通行距离分布
plt.subplot(1, 2, 1)
sns.histplot(df['通行距离'], bins=30, kde=True, color='skyblue')
plt.axvline(x=std_distance, color='red', linestyle='--', label=f'国际标准({std_distance}m)')
plt.title('小区到地铁站通行距离分布')
plt.xlabel('通行距离(米)')
plt.ylabel('小区数量')
plt.legend()

# 超标情况
plt.subplot(1, 2, 2)
exceed = (df['通行距离'] > std_distance).value_counts(normalize=True) * 100
print(exceed)
plt.pie(exceed, labels=['超标','达标'], autopct='%1.1f%%',
        colors=['salmon', 'lightgreen'], explode=(0.1, 0))
plt.title(f'通行距离达标情况(>{std_distance}m)')

plt.tight_layout()
plt.savefig('站通行距离分布及超标情况.png', dpi=400, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 6))

# 步行时间分布
plt.subplot(1, 2, 1)
sns.boxplot(y=df['步行时间（分钟）'], color='lightblue')
plt.axhline(y=std_time, color='red', linestyle='--', label=f'标准时间({std_time}分钟)')
plt.title('步行时间分布')
plt.ylabel('步行时间(分钟)')
plt.legend()

# 步行时间与距离关系
# plt.subplot(1, 2, 2)
# sns.regplot(x='通行距离', y='步行时间（分钟）', data=df, color='blue')
# plt.title('通行距离与步行时间关系')
# plt.xlabel('通行距离(米)')
# plt.ylabel('步行时间(分钟)')
plt.subplot(1, 2, 2)
sns.regplot(
    x='通行距离',
    y='步行时间（分钟）',
    data=df,
    scatter_kws={'color': 'green', 'alpha': 0.6},  # 设置散点颜色为绿色
    line_kws={'color': 'red', 'linewidth': 2}     # 设置回归线颜色为红色
)
plt.title('通行距离与步行时间关系')
plt.xlabel('通行距离(米)')
plt.ylabel('步行时间(分钟)')


plt.tight_layout()
plt.savefig('步行时间分布.png', dpi=400, bbox_inches='tight')
plt.show()


plt.figure(figsize=(16, 12), dpi=400)

# 不同地铁站的通行距离比较
plt.subplot(2, 1, 1)
sns.boxplot(x='最近地铁站', y='通行距离', data=df, palette='pastel')
plt.axhline(y=std_distance, color='red', linestyle='--', label='国际标准')
plt.title('不同地铁站的通行距离比较')
plt.xticks(rotation=90, fontsize=8)
plt.legend()

# 不同地铁站的步行时间比较
plt.subplot(2, 1, 2)
sns.violinplot(x='最近地铁站', y='步行时间（分钟）', data=df, palette='pastel')
plt.axhline(y=std_time, color='red', linestyle='--', label='国际标准')
plt.title('不同地铁站的步行时间比较')
plt.xticks(rotation=90, fontsize=8)
plt.legend()

plt.tight_layout()
plt.savefig('通行距离比较_步行时间比较1.png', dpi=400, bbox_inches='tight')
plt.show()


# 计算通行距离与步行时间的相关性
corr = df[['直线距离', '通行距离', '步行时间（分钟）', '骑行时间（分钟）', '驾车（分钟）']].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('各交通指标相关性热力图', pad=20)  # 增加标题与图的间距
plt.tight_layout()  # 自动调整子图参数，避免标签被截断
plt.savefig('通行距离与步行时间的相关性.png', dpi=400, bbox_inches='tight')
plt.show()


# 模拟地理位置可视化
np.random.seed(42)
df['模拟经度'] = np.random.normal(116.3, 0.1, size=len(df))
df['模拟纬度'] = np.random.normal(39.9, 0.05, size=len(df))

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['模拟经度'], df['模拟纬度'],
                     c=df['通行距离'], cmap='viridis', s=df['步行时间（分钟）']*10, alpha=0.7)

# 添加地铁站位置
metro_stations = {
    '六里桥地铁站': (116.3, 39.88),
    '莲花桥地铁站': (116.31, 39.92),
    '五棵松地铁站': (116.27, 39.91)
}

for name, (lon, lat) in metro_stations.items():
    plt.scatter(lon, lat, marker='*', s=200, color='red', label=f'{name}')

plt.colorbar(scatter, label='通行距离(米)')
plt.legend()
plt.title('小区与地铁站的空间分布(模拟)')
plt.xlabel('经度')
plt.ylabel('纬度')

# 添加图例说明气泡大小
for time in [10, 15, 20]:
    plt.scatter([], [], c='k', alpha=0.3, s=time*10, label=f'{time}分钟')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='步行时间')
plt.savefig('模拟地理位置可视化.png', dpi=400, bbox_inches='tight')
plt.show()


# 定义超标等级
df['超标等级'] = pd.cut(df['超标准距离'],
                      bins=[-np.inf, 0, 400, 800, np.inf],
                      labels=['达标','轻度超标','中度超标','严重超标'])

plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='超标等级', order=df['超标等级'].value_counts().index,
             palette='RdYlGn_r')
plt.title('小区地铁可达性超标等级分布', fontsize=16)
plt.xlabel('超标等级')
plt.ylabel('小区数量')

for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.0f}\n({p.get_height()/len(df):.1%})',
                      (p.get_x()+p.get_width()/2, p.get_height()+10),
                      ha='center')
plt.savefig('超标等级.png', dpi=400, bbox_inches='tight')
plt.show()