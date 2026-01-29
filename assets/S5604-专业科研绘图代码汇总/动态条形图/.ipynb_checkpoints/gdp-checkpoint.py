import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 创建示例GDP数据
data = {
    'Country': ['USA', 'China', 'Japan', 'Germany', 'India'],
    '2015': [18036, 11226, 4380, 3362, 2104],
    '2016': [18569, 11559, 4938, 3478, 2295],
    '2017': [19390, 12327, 4850, 3681, 2660],
    '2018': [20580, 13407, 4971, 3948, 2713],
    '2019': [21433, 14342, 5081, 4137, 2875],
}

# 转换为DataFrame
df = pd.DataFrame(data)
df = df.set_index('Country').T  # 转置数据方便处理

# 设置图形大小和样式
fig, ax = plt.subplots(figsize=(10, 6))
# plt.style.use('seaborn-pastel')
plt.style.use('ggplot') 

def update(frame):
    ax.clear()
    year = df.index[frame]
    ax.barh(df.columns, df.iloc[frame], color='skyblue')
    ax.set_title(f'Global GDP Ranking in {year}', fontsize=16)
    ax.set_xlabel('GDP in Billion USD')
    ax.set_xlim(0, df.max().max() + 5000)
    ax.set_ylabel('Country')
    for i, (value, name) in enumerate(zip(df.iloc[frame], df.columns)):
        ax.text(value, i, f'{value:.0f}', va='center')

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(df), interval=1000, repeat=False)

# 保存动画为GIF或显示
ani.save('gdp_ranking.gif', writer='imagemagick')
plt.show()
