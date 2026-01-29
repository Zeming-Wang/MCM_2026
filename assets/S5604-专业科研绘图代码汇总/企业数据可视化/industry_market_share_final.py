import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Pie

# 由于数据错误，行业列里面是省份名称，需要排除的省份列表
exclude_provinces = ["北京", "广东", "山东", "四川", "安徽", "江苏", "辽宁", "宁夏",
                     "浙江", "湖北", "陕西", "重庆", "内蒙古", "天津", "云南", "上海"]

# 读取数据
df = pd.read_csv('Alldata.csv', encoding='utf-8')

# 数据清洗
# 删除关键字段为空的数据
df = df.dropna(subset=['市值', '行业'])

# 删除行业列中包含省份名称的数据行
province_mask = df['行业'].apply(lambda x: any(province in str(x) for province in exclude_provinces))
print(f"\n发现市值列包含省份的记录数: {province_mask.sum()}")
print("将被删除的示例数据:")
print(df[province_mask].head()[['公司全称', '市值']])

df = df[~province_mask].copy()
print(f"删除后剩余数据量: {len(df)}")


# 市值单位转换函数
def convert_market_value(value):
    try:
        # 提取数字部分
        num_str = ''.join([c for c in str(value).split('：')[1] if c.isdigit() or c == '.'])
        num = float(num_str)

        if '亿' in str(value):
            return num
        elif '万' in str(value):
            return num / 10000
        elif '美元' in str(value):
            return num * 7 / 10000 if '万' in str(value) else num * 7
        return num  # 默认按亿处理
    except:
        print(f"无法解析的市值值: {value}")
        return None


df['市值_亿'] = df['市值'].apply(convert_market_value)
df = df.dropna(subset=['市值_亿'])

# 按行业统计，使用清洗后的行业列
industry_data = (
    df.groupby('行业', as_index=False)['市值_亿']
    .sum()
    .sort_values('市值_亿', ascending=False)
)

# 绘制饼图
data_pair = [(row['行业'], round(row['市值_亿'], 2)) for _, row in industry_data.iterrows()]

pie = (
    Pie(init_opts=opts.InitOpts(width="1000px", height="600px"))
    .add(
        series_name="市值(亿元)",
        data_pair=data_pair,
        radius=["0%", "65%"],
        center=["50%", "50%"],
        label_opts=opts.LabelOpts(
            formatter="{b}|{c}亿|{d}%",
            font_size=12,
            rich={
                "b": {"fontSize": 14, "lineHeight": 33},
                "per": {"color": "#eee", "backgroundColor": "#334455"}
            }
        )
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="各行业市值占比分析",
            pos_left="center",
            title_textstyle_opts=opts.TextStyleOpts(font_size=20)
        ),
        legend_opts=opts.LegendOpts(
            orient="vertical",
            pos_right="5%",
            pos_top="15%",
            item_width=40,
            item_height=20
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter="{a}<br/>{b}: {c}亿元 ({d}%)"
        )
    )
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(trigger="item"),
        label_opts=opts.LabelOpts(
            position="outside",
            formatter="{b}: {d}%",
        )
    )
)

# 保存结果
output_file = "industry_market_share_final.html"
pie.render(output_file)
print(f"\n图表已生成：{output_file}")
print("\n最终行业分布：")
print(industry_data)