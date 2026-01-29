import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Pie
import re

# 读取CSV文件
df = pd.read_csv('Alldata.csv')

# 处理数据：删除企业类型为空的行
df = df.dropna(subset=['类型'])
df = df[
    ~df['类型'].isin(['', '--', '110000005118165', '未公开']) &  # 排除空字符串和'--'
    df['类型'].notna()  # 排除NaN值
]
# 清洗企业类型字段：去除括号及括号内容
def clean_company_type(type_str):
    # 使用正则表达式匹配并去除括号及括号内容
    cleaned_type = re.sub(r'[（(].*?[)）]', '', type_str)
    return cleaned_type.strip()

df['清洗后类型'] = df['类型'].apply(clean_company_type)

# 统计各类型数量
type_counts = df['清洗后类型'].value_counts()

# 准备环形图数据
data = [(name, count) for name, count in type_counts.items()]

# 创建环形图
c = (
    Pie(init_opts=opts.InitOpts(width="2400px", height="1200px"))
    .add(
        "",
        data,
        radius=["25%", "50%"],  # 内半径和外半径，形成环形效果
        center=["50%", "55%"],  # 整体下移5%
        label_opts=opts.LabelOpts(
            position="outside",
            formatter="{b}: {c} ({d}%)",  # 显示名称、数量和百分比
        ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="企业类型占比环形图",
                                  pos_left="center",    # 标题基准定位在右侧
                                  pos_top="1%",
                                 ),
        legend_opts=opts.LegendOpts(
            orient="vertical",  # 垂直排列
            pos_right="15%",     # 距离右侧5%的位置
            pos_top="center",   # 垂直居中
            item_width=20,      # 图例标记的图形宽度
            item_height=10,     # 图例标记的图形高度
            item_gap=5          # 图例每项之间的间隔
        ),
    )
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
    )
)

# 渲染图表到HTML文件
c.render("company_type_pie_chart.html")