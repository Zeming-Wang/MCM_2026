import pandas as pd
from pyecharts.charts import Pie
from pyecharts import options as opts

## 读取CSV文件
df = pd.read_csv('Alldata.csv')

## 数据清洗
# 删除行业为空值的数据
df = df.dropna(subset=['行业'])

# 由于数据错误，行业列里面是省份名称，需要排除的省份列表
exclude_provinces = ["北京", "广东", "山东", "四川", "安徽", "江苏", "辽宁", "宁夏",
                    "浙江", "湖北", "陕西", "重庆", "内蒙古", "天津", "云南", "上海"]

# 删除行业列中包含在排除列表中的数据
df = df[~df['行业'].isin(exclude_provinces)]

# 按行业分组计算企业数量
industry_counts = df.groupby('行业').size().reset_index(name='数量')
# 排序或者乱序
# industry_counts = industry_counts.sample(frac=1).reset_index(drop=True)
industry_counts = industry_counts.sort_values(by='数量').reset_index(drop=True)
print(industry_counts)
# 准备数据
data_pair = list(industry_counts[['行业', '数量']].itertuples(index=False, name=None))

# 绘制环形图
pie = Pie(init_opts=opts.InitOpts(width="1200px", height="600px"))

pie.add(
    series_name="各行业企业数量占比",
    data_pair=data_pair,  # 指定数据
    radius=["40%", "75%"],  # 内半径40%，外半径75%，形成环形
    label_opts=opts.LabelOpts(
        formatter="{b}: {d}%",
    ),  # 显示百分比数据
    # rosetype="radius"  # 若需要玫瑰图效果可取消注释
)


pie.set_global_opts(
    title_opts=opts.TitleOpts(title="各行业企业数量占比环形图", pos_left="center",),
    legend_opts=opts.LegendOpts(orient="vertical",
        pos_right="1%",       # 距离右侧5%的位置
        pos_top="bottom",     # 垂直居中
        item_width=14,        # 图例标记宽度
        item_height=14,        # 图例标记高度
    )
)

pie.render("industry_company_count_pie.html")