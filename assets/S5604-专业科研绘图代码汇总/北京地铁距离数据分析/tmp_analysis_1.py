import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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

# 计算与国际标准(800米)的差距
df['超标准距离'] = df['通行距离'] - 800
df['超标准时间'] = df['步行时间（分钟）'] - 10  # 国际标准12分钟


# plt.style.use('seaborn')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 通行距离分布
sns.histplot(data=df, x='通行距离', bins=30, kde=True, ax=axes[0,0], color='royalblue')
axes[0,0].axvline(800, color='red', linestyle='--', linewidth=2)
axes[0,0].set_title('通行距离分布（国际标准800米）', fontsize=14)
axes[0,0].set_xlabel('通行距离（米）')
axes[0,0].annotate(f'超标比例：{(df["通行距离"]>800).mean():.1%}',
                  xy=(0.7, 0.9), xycoords='axes fraction')

# 步行时间分布
sns.boxplot(data=df, y='步行时间（分钟）', ax=axes[0,1], color='salmon')
axes[0,1].axhline(12, color='green', linestyle='--', linewidth=2)
axes[0,1].set_title('步行时间分布（国际标准12分钟）', fontsize=14)

# 超标程度分布
sns.ecdfplot(data=df, x='超标准距离', ax=axes[1,0], color='purple')
axes[1,0].axvline(0, color='red', linestyle='--')
axes[1,0].set_title('累计超标距离分布', fontsize=14)
axes[1,0].set_xlabel('超出标准距离（米）')

# 超标时间热区
hexbin = axes[1,1].hexbin(df['通行距离'], df['步行时间（分钟）'],
                         gridsize=30, cmap='YlOrRd', bins='log')
plt.colorbar(hexbin, ax=axes[1,1])
axes[1,1].axhline(12, color='black', linestyle='--')
axes[1,1].axvline(800, color='black', linestyle='--')
axes[1,1].set_title('通行距离-步行时间热力图', fontsize=14)

plt.tight_layout()
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
plt.show()


# 生成模拟地理坐标
np.random.seed(42)
df['经度'] = 116.25 + np.random.rand(len(df)) * 0.5
df['纬度'] = 39.8 + np.random.rand(len(df)) * 0.4

plt.figure(figsize=(12, 10))
sc = plt.scatter(df['经度'], df['纬度'],
                c=df['通行距离'], cmap='viridis_r',
                s=df['步行时间（分钟）']*2, alpha=0.7)
plt.colorbar(sc, label='通行距离（米）')

# 添加地铁站标记
stations = {'六里桥':(116.3,39.88), '国贸':(116.46,39.91), '西直门':(116.35,39.94)}
for name, coord in stations.items():
    plt.scatter(*coord, marker='*', s=300, color='red')
    plt.text(coord[0]+0.01, coord[1], name, fontsize=12)

# 创建图例
for size in [20, 40, 60]:
    plt.scatter([], [], c='gray', alpha=0.5, s=size,
               label=f'{size//2}分钟')
plt.legend(title='步行时间', loc='upper right')

plt.title('北京小区地铁可达性空间分布（模拟）', fontsize=16)
plt.xlabel('经度')
plt.ylabel('纬度')
plt.grid(True, alpha=0.3)
plt.savefig('北京小区地铁可达性空间分布（模拟）.png', dpi=400, bbox_inches='tight')
plt.show()


# 计算接驳公交潜在效益
df['接驳节省时间'] = (df['步行时间（分钟）'] - 5).clip(lower=0)  # 假设接驳将时间缩短至5分钟

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 节省时间分布
sns.histplot(df['接驳节省时间'], bins=20, ax=axes[0], color='teal')
axes[0].set_title('接驳公交潜在节省时间分布', fontsize=14)
axes[0].set_xlabel('可节省时间（分钟）')

# 节省时间与距离关系
sns.regplot(data=df, x='通行距离', y='接驳节省时间',
           scatter_kws={'alpha':0.3}, line_kws={'color':'red'},
           ax=axes[1])
axes[1].set_title('通行距离与可节省时间关系', fontsize=14)
axes[1].set_xlabel('通行距离（米）')
axes[1].set_ylabel('可节省时间（分钟）')

plt.tight_layout()
plt.show()


# 计算接驳收益成本比
plt.figure(figsize=(12, 6))
cost_per_km = 0.3  # 假设每公里运营成本0.3元
df['接驳收益'] = df['接驳节省时间'] * 0.5  # 假设每分钟时间价值0.5元
df['接驳成本'] = df['通行距离']/1000 * cost_per_km
df['收益成本比'] = df['接驳收益'] / df['接驳成本']

sns.kdeplot(data=df, x='收益成本比', fill=True, color='green')
plt.axvline(1, color='red', linestyle='--')
plt.title('接驳公交经济性分析（收益成本比）', fontsize=16)
plt.xlabel('收益/成本比率')
plt.ylabel('密度')
plt.text(1.5, 0.5, f"盈利比例：{(df['收益成本比']>1).mean():.1%}",
        fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
plt.grid(True, alpha=0.3)
plt.show()