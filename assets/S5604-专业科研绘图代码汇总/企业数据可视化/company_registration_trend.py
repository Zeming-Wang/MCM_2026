import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
from datetime import datetime

# 读取CSV文件
df = pd.read_csv('Alldata.csv', encoding='utf-8')

# 数据清洗
def parse_registration_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except:
        return None

df['注册时间'] = df['注册时间'].apply(parse_registration_date)
df = df.dropna(subset=['注册时间'])

# 筛选2000-2010年的数据
df = df[(df['注册时间'] >= '2000-01-01') & (df['注册时间'] <= '2010-12-31')]

# 按年份统计注册数量
df['注册年份'] = df['注册时间'].dt.year
yearly_counts = df['注册年份'].value_counts().sort_index()

# 补全缺失年份
all_years = range(2000, 2011)
yearly_counts = yearly_counts.reindex(all_years, fill_value=0)

print("2000-2010年企业注册数量统计：")
print(yearly_counts)

# 绘制折线图
line = (
    Line(init_opts=opts.InitOpts(width="1000px", height="500px"))
    .add_xaxis(xaxis_data=[str(y) for y in all_years])  # 确保x轴为字符串类型
    .add_yaxis(
        series_name="企业注册数量",
        y_axis=yearly_counts.tolist(),
        is_smooth=True,
        symbol="circle",
        symbol_size=8,
        label_opts=opts.LabelOpts(is_show=True),
        linestyle_opts=opts.LineStyleOpts(width=3),
        itemstyle_opts=opts.ItemStyleOpts(
            border_width=2,
            border_color="#fff",
            color="#c23531"
        )
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="2000-2010年企业注册数量变化趋势",
            subtitle=f"总企业数: {yearly_counts.sum()}家",
            pos_left="center"
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="axis",
            formatter="{a}<br/>{b}: {c}家"
        ),
        xaxis_opts=opts.AxisOpts(
            name="年份",
            type_="category",  # 明确指定为类别轴
            boundary_gap=False
        ),
        yaxis_opts=opts.AxisOpts(
            name="注册数量",
            type_="value",
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        legend_opts=opts.LegendOpts(
            pos_right="4%",
            pos_top="15%",
            orient="vertical",
            item_width=50,
            item_height=20,
            textstyle_opts=opts.TextStyleOpts(font_size=12)
        ),
    )
)

# 保存并显示
output_file = "company_registration_trend.html"
line.render(output_file)
print(f"\n图表已保存至: {output_file}")