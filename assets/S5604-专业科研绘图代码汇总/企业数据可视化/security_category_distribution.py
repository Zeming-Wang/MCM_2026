import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar

# 读取CSV文件
df = pd.read_csv('Alldata.csv', encoding='utf-8')

# 数据清洗
# 处理证券类别字段
df['证券类别'] = df['证券类别'].str.strip()  # 去除前后空格
# 删除无效数据（空值或'--'）
df = df[
    ~df['证券类别'].isin(['', '--']) &  # 排除空字符串和'--'
    df['证券类别'].notna()  # 排除NaN值
]

print(df[:5])
# 统计各证券类别的数量
category_counts = df['证券类别'].value_counts().reset_index()
category_counts.columns = ['证券类别', '数量']

# 绘制柱形图
bar = (
    Bar(init_opts=opts.InitOpts(width="1200px", height="600px"))
    .add_xaxis(category_counts['证券类别'].tolist())
    .add_yaxis(
        "企业数量",
        category_counts['数量'].tolist(),
        category_gap="60%",  # 柱子间距
        itemstyle_opts=opts.ItemStyleOpts(color="#5470C6")  # 柱子颜色
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="企业证券类别数量分布",
            subtitle=f"有效数据量: {len(df)}条",
            pos_left="center"
        ),
        xaxis_opts=opts.AxisOpts(
            name="证券类别",
            axislabel_opts=opts.LabelOpts(rotate=45)  # x轴标签旋转45度防重叠
        ),
        yaxis_opts=opts.AxisOpts(
            name="企业数量",
            splitline_opts=opts.SplitLineOpts(is_show=True)  # 显示横向网格线
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="axis",
            formatter="{b}<br/>企业数量: {c}"
        ),
        legend_opts=opts.LegendOpts(
            orient="vertical",
            pos_right="5%",
            pos_top="15%",
            item_width=40,
            item_height=20
        ),
    )
    .set_series_opts(
        label_opts=opts.LabelOpts(
            position="top",
            formatter="{c}"  # 在柱子上方显示数值
        )
    )
)

# 保存图表
output_file = "security_category_distribution.html"
bar.render(output_file)
print(f"图表已生成: {output_file}")
print("\n证券类别统计结果:")
print(category_counts)