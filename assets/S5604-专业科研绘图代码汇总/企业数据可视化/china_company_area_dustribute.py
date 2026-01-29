import pandas as pd
from pyecharts.charts import Map
from pyecharts import options as opts

df = pd.read_csv('Alldata.csv', encoding='utf-8')
exclude_provinces = ["海外"]
# 数据清洗
# 删除地址为空或NaN的数据
df = df.dropna(subset=['地址'])
df = df[df['地址'] != '']
province_mask = df['地址'].apply(lambda x: any(province in str(x) for province in exclude_provinces))
df = df[~province_mask].copy()

# 统计各省份企业数量
province_counts = df['地址'].value_counts().reset_index()
province_counts.columns = ['省份', '企业数量']

# 准备热力图数据（格式：[(省份, 数量), ...]）
heatmap_data = [(prov, count) for prov, count in zip(province_counts['省份'], province_counts['企业数量'])]

province_mapping = {
    '北京': '北京市',
    '广东': '广东省',
    '上海': '上海市',
    '浙江': '浙江省',
    '江苏': '江苏省',
    '山东': '山东省',
    '福建': '福建省',
    '四川': '四川省',
    '湖北': '湖北省',
    '河南': '河南省',
    '湖南': '湖南省',
    '安徽': '安徽省',
    '辽宁': '辽宁省',
    '重庆': '重庆市',
    '河北': '河北省',
    '天津': '天津市',
    '陕西': '陕西省',
    '江西': '江西省',
    '广西': '广西壮族自治区',
    '云南': '云南省',
    '黑龙江': '黑龙江省',
    '海南': '海南省',
    '山西': '山西省',
    '吉林': '吉林省',
    '贵州': '贵州省',
    '新疆': '新疆维吾尔自治区',
    '内蒙古': '内蒙古自治区',
    '甘肃': '甘肃省',
    '宁夏': '宁夏回族自治区',
    '西藏': '西藏自治区',
    '青海': '青海省',
    '香港': '香港特别行政区'
}
# 转换为全称
converted_data = [(province_mapping[prov], count) for prov, count in heatmap_data]
print(converted_data)

# 创建地图实例
china_map = Map(init_opts=opts.InitOpts(width="1300px", height="1000px"))

# 添加数据（maptype="china"）
china_map.add(
    series_name="企业数量",
    data_pair=converted_data,
    maptype="china",
    is_map_symbol_show=False,
    label_opts=opts.LabelOpts(is_show=True)  # 显示省份标签
)

# 设置全局配置
china_map.set_global_opts(
    title_opts=opts.TitleOpts(title="中国企业地域分布",
                              pos_left="center",
                              pos_top="1%",       # 标题上移
                              padding=[0, 0, 30, 0]  # 增加下边距
                             ),
    visualmap_opts=opts.VisualMapOpts(
        min_=0,
        max_=2600,
        range_text=["低", "高"],
        is_piecewise=True,
        pos_left="center",
        pos_bottom="5%",
        textstyle_opts=opts.TextStyleOpts(color="#000"),
    ),
    legend_opts=opts.LegendOpts(pos_top="8%"),  # 图例下移
)

# 渲染为 HTML
china_map.render("china_company_area_dustribute.html")

print("地域分布地图已生成")