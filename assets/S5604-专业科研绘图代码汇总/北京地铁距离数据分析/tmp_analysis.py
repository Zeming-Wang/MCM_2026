import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建DataFrame
data = {
    '序号': range(1, 31),
    '名称': ['1点', '2点', '3点', '4点', '5点', '6点', '7点', '8点', '9点', '10点',
           '11点', '12点', '13点', '14点', '15点', '16点', '17点', '18点', '19点', '20点',
           '21点', '22点', '23点', '24点', '25点', '26点', '27点', '28点', '29点', '30点'],
    '地址': ['万丰路一十八号', '莲花路10号院', '万丰路一十八号', '莲花路10号院', '莲花路10号院',
           '莲花路10号院', '莲花路10号院', '海淀九号', '吴家场经济小区', '海淀九号',
           '海淀九号', '吴家场经济小区', '莲花路七号院', '莲花小区', '莲花小区', '莲花小区',
           '莲花小区', '莲花小区', '吴家场经济适用月日', '莲花小区', '莲花小区', '莲花小区',
           '吴家场经济适用月日', '太平路44号院', '新兴华中区', '太平路44号院', '太平路甲21号院',
           '太平路44号院', '太平路44号院', '太平路44号院'],
    '方位': ['南1', '南2', '北3', '东4', '东5', '北6', '南7', '南8', '南9', '南10',
           '北11', '南12', '北13', '南14', '南15', '南16', '东17', '南18', '北19', '北20',
           '北21', '北22', '南23', '南24', '南25', '南26', '东27', '东28', '北29', '北30'],
    '最近地铁站': ['六里桥地铁站']*11 + ['莲花桥地铁站']*8 + ['五棵松地铁站']*7 + ['莲花桥地铁站']*4,
    '直线距离': [798.75, 791.62, 945.06, 804.00, 843.85, 934.61, 895.71, 962.39, 985.40, 988.31,
              1011.97, 935.16, 1036.88, 496.08, 289.32, 664.01, 244.68, 1056.65, 520.60, 410.73,
              419.77, 677.84, 1262.93, 1120.58, 1215.06, 1018.64, 1030.57, 861.92, 1043.95, 1043.95],
    '通行距离': [1089.63, 1100.21, 1330.12, 1097.46, 1069.86, 1253.43, 1097.99, 1260.19, 963.30, 1226.13,
              1423.68, 1194.86, 1342.93, 1049.52, 1288.61, 784.21, 216.55, 1138.28, 579.99, 529.62,
              478.03, 764.69, 1718.02, 1449.12, 1711.07, 1391.56, 1707.83, 925.34, 1588.31, 1588.31],
    '步行时间': [18.16, 19.34, 22.17, 18.29, 17.83, 20.89, 18.30, 21.00, 16.05, 20.48,
              23.72, 19.91, 22.38, 17.49, 21.48, 18.07, 6.74, 18.97, 9.67, 8.83,
              7.97, 12.74, 28.64, 8.12, 28.53, 23.20, 28.46, 15.42, 26.44, 26.44],
    '骑行时间': [3.63, 3.69, 4.43, 3.66, 3.57, 4.18, 3.66, 4.20, 3.21, 4.09,
              4.75, 3.98, 4.48, 3.50, 4.29, 2.61, 1.32, 2.79, 1.93, 1.77,
              1.59, 2.55, 5.73, 4.83, 5.70, 4.64, 5.69, 3.08, 5.29, 5.29],
    '驾车时间': [1.25, 1.54, 1.37, 1.61, 1.68, 1.47, 1.61, 1.48, 1.70, 1.73,
              1.67, 1.78, 2.04, 1.07, 1.23, 0.90, 0.55, 1.30, 0.57, 0.47,
              0.37, 0.90, 2.21, 2.06, 1.91, 2.42, 2.35, 1.83, 2.10, 2.10]
}

df = pd.DataFrame(data)

# 计算与国际标准(800米)的差距
df['超标准距离'] = df['通行距离'] - 800
df['超标准时间'] = df['步行时间'] - 12  # 国际标准12分钟

# 基本统计
print(df[['通行距离', '步行时间', '超标准距离', '超标准时间']].describe())

plt.figure(figsize=(12, 6))

# 通行距离分布
plt.subplot(1, 2, 1)
sns.histplot(df['通行距离'], bins=10, kde=True, color='skyblue')
plt.axvline(x=800, color='red', linestyle='--', label='国际标准(800m)')
plt.title('小区到地铁站通行距离分布')
plt.xlabel('通行距离(米)')
plt.ylabel('小区数量')
plt.legend()

# 超标情况
plt.subplot(1, 2, 2)
exceed = (df['通行距离'] > 800).value_counts(normalize=True) * 100
plt.pie(exceed, labels=['达标', '超标'], autopct='%1.1f%%',
        colors=['lightgreen', 'salmon'], explode=(0.1, 0))
plt.title('通行距离达标情况(>800m)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# 步行时间分布
plt.subplot(1, 2, 1)
sns.boxplot(y=df['步行时间'], color='lightblue')
plt.axhline(y=12, color='red', linestyle='--', label='国际标准(12分钟)')
plt.title('步行时间分布')
plt.ylabel('步行时间(分钟)')
plt.legend()

# 步行时间与距离关系
plt.subplot(1, 2, 2)
sns.regplot(x='通行距离', y='步行时间', data=df, color='green')
plt.title('通行距离与步行时间关系')
plt.xlabel('通行距离(米)')
plt.ylabel('步行时间(分钟)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

# 不同地铁站的通行距离比较
plt.subplot(1, 2, 1)
sns.boxplot(x='最近地铁站', y='通行距离', data=df, palette='pastel')
plt.axhline(y=800, color='red', linestyle='--', label='国际标准')
plt.title('不同地铁站的通行距离比较')
plt.xticks(rotation=15)
plt.legend()

# 不同地铁站的步行时间比较
plt.subplot(1, 2, 2)
sns.violinplot(x='最近地铁站', y='步行时间', data=df, palette='coolwarm')
plt.axhline(y=12, color='red', linestyle='--', label='国际标准')
plt.title('不同地铁站的步行时间比较')
plt.xticks(rotation=15)
plt.legend()

plt.tight_layout()
plt.show()

# 计算通行距离与步行时间的相关性
corr = df[['直线距离', '通行距离', '步行时间', '骑行时间', '驾车时间']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('各交通指标相关性热力图')
plt.show()

# 模拟地理位置可视化
np.random.seed(42)
df['模拟经度'] = np.random.normal(116.3, 0.1, size=len(df))
df['模拟纬度'] = np.random.normal(39.9, 0.05, size=len(df))

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['模拟经度'], df['模拟纬度'],
                     c=df['通行距离'], cmap='viridis', s=df['步行时间']*10, alpha=0.7)

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

plt.show()

