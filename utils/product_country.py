"""
Product Description & Country Analysis Module

商品描述 (Description) 与用户所在国家 (Country) 两个特征的业务探索。
纯计算模块, 不含任何 Streamlit 代码。

关于两个特征的分析定位 (见说明文档):
  - Description 是文本字段, 4,632 种商品, 无法直接做统计 —— 这里用「描述里的高频词」
    把商品归成可解释的品类 (如 CAKE CASE / T-LIGHT HOLDER / LUNCH BAG), 再看各品类的表现
  - Country 是类别字段且分布极度集中 (英国占 91% 的行、84.7% 的收入), 方差太小,
    不适合作为 K-Means 的输入特征; 但作为分析维度差异明显, 因此用于「本土 vs 海外」画像对比
"""
import re
import pandas as pd
import numpy as np

# 描述里的功能词/泛化词: 频率高但没有品类区分度, 统计品类时排除
# (刻意保留 BAG / HOLDER / CABINET 这类真正指向品类的词)
_STOPWORDS = {
    'PACK', 'SET', 'WITH', 'AND', 'THE', 'FOR', 'OF', 'NEW', 'DESIGN',
    'PLUS', 'TOTAL', 'STYLE', 'COLOUR', 'COLOR', 'ONE', 'TWO', 'THREE',
}


def _extract_words(description: str) -> set:
    """从商品描述里提取候选品类词 (全大写、长度 >= 3、非功能词)。"""
    words = set()
    for raw in re.findall(r"[A-Z][A-Z'\-]{2,}", str(description).upper()):
        w = raw.strip("'-")
        if len(w) >= 3 and w not in _STOPWORDS:
            words.add(w)
    return words


# 高频词的中文注释: cn = 中文含义, type = 词类
# 逐词核对过实际商品名 (取该词收入最高的几条) 后填写, 不是凭词义猜的, 例如:
#   HOT / WATER / BOTTLE 三个词其实都来自同一个商品名 "HOT WATER BOTTLE" (热水袋),
#   T-LIGHT 来自 "T-LIGHT HOLDER", UNION 来自 "UNION JACK" (英国国旗图案)
WORD_CN = {
    # 颜色类
    'RED':      {'cn': '红色系列', 'type': '颜色'},
    'WHITE':    {'cn': '白色系列', 'type': '颜色'},
    'PINK':     {'cn': '粉色系列', 'type': '颜色'},
    'BLUE':     {'cn': '蓝色系列', 'type': '颜色'},
    'BLACK':    {'cn': '黑色系列', 'type': '颜色'},
    # 图案类
    'RETROSPOT': {'cn': '复古圆点图案', 'type': '图案'},
    'SPOTTY':   {'cn': '圆点图案', 'type': '图案'},
    'SPOT':     {'cn': '圆点图案 (RETRO SPOT 系列)', 'type': '图案'},
    'UNION':    {'cn': '英国国旗图案 (Union Jack)', 'type': '图案'},
    # 风格类
    'VINTAGE':  {'cn': '复古风格', 'type': '风格'},
    'RETRO':    {'cn': '复古风格', 'type': '风格'},
    # 品类类
    'BAG':      {'cn': '袋类 (购物袋 / 手提袋)', 'type': '品类'},
    'HOLDER':   {'cn': '托架类 (烛台 / 卡片架)', 'type': '品类'},
    'CAKE':     {'cn': '蛋糕装饰 (纸杯 / 托盘)', 'type': '品类'},
    'BOX':      {'cn': '盒类 (首饰盒 / 收纳盒)', 'type': '品类'},
    'T-LIGHT':  {'cn': '茶蜡托 / 小蜡烛', 'type': '品类'},
    'SIGN':     {'cn': '标牌 / 挂牌', 'type': '品类'},
    'TEA':      {'cn': '茶具 / 茶主题', 'type': '品类'},
    'HOT':      {'cn': '热水袋 (来自 HOT WATER BOTTLE)', 'type': '品类'},
    'BOTTLE':   {'cn': '热水袋 / 瓶类', 'type': '品类'},
    'WATER':    {'cn': '热水袋 (来自 HOT WATER BOTTLE)', 'type': '品类'},
    'LUNCH':    {'cn': '午餐袋系列', 'type': '品类'},
    # 材质类
    'METAL':    {'cn': '金属材质 (多为金属标牌)', 'type': '材质'},
    'PAPER':    {'cn': '纸质品 (拉花 / 纸链)', 'type': '材质'},
    'GLASS':    {'cn': '玻璃材质', 'type': '材质'},
    # 形态 / 规格 / 主题
    'HEART':    {'cn': '心形造型', 'type': '形态'},
    'HANGING':  {'cn': '悬挂式', 'type': '形态'},
    'JUMBO':    {'cn': '超大规格', 'type': '规格'},
    'SMALL':    {'cn': '小规格', 'type': '规格'},
    'CHRISTMAS': {'cn': '圣诞主题', 'type': '主题'},
}


def _word_cn(word: str) -> dict:
    """取词的中文注释, 未登记的词返回占位, 保证页面不会出现 None。"""
    return WORD_CN.get(word, {'cn': '—', 'type': '其他'})


def description_word_stats(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    按商品描述里的高频词统计品类表现 (数据驱动, 不依赖人工品类词表)。

    做法: 先把交易聚合到商品 (StockCode + Description) 粒度, 再让每个商品匹配它描述
    里包含的候选词, 最后汇总。

    注意口径: 一个商品可能命中多个词 (如 "JUMBO BAG PINK WITH WHITE SPOTS" 同时含
    JUMBO / BAG / PINK / SPOTS), 因此各关键词的合计会互相重叠, 不能相加当作总量。

    Parameters:
        df: 清洗后交易数据
        top_n: 返回收入最高的前 N 个品类词

    Returns:
        DataFrame: 关键词, 中文含义, 词类, 显示名, 商品数, 收入, 收入占比%, 销量, 平均单价
    """
    items = df.groupby(['StockCode', 'Description']).agg(
        收入=('Revenue', 'sum'),
        销量=('Quantity', 'sum'),
    ).reset_index()

    acc = {}
    for row in items.itertuples(index=False):
        for w in _extract_words(row.Description):
            a = acc.setdefault(w, {'商品数': 0, '收入': 0.0, '销量': 0})
            a['商品数'] += 1
            a['收入'] += float(row.收入)
            a['销量'] += int(row.销量)

    out = pd.DataFrame([{'关键词': k, **v} for k, v in acc.items()])
    if out.empty:
        return out

    total_revenue = float(df['Revenue'].sum())
    out['收入占比%'] = (out['收入'] / total_revenue * 100).round(2)
    out['平均单价'] = (out['收入'] / out['销量']).round(2)
    out['收入'] = out['收入'].round(0)

    # 中文注释与词类: 屏幕上只有英文品类词对观众不友好, 这里补上中文含义;
    # 词类同时支撑"这些高频词以颜色/图案/风格类为主"这个结论 (见第 6 页解读)
    out['中文含义'] = out['关键词'].map(lambda w: _word_cn(w)['cn'])
    out['词类'] = out['关键词'].map(lambda w: _word_cn(w)['type'])
    out['显示名'] = out['关键词'] + ' · ' + out['中文含义']

    out = out.sort_values('收入', ascending=False).head(top_n).reset_index(drop=True)
    return out[['关键词', '中文含义', '词类', '显示名', '商品数', '收入',
                '收入占比%', '销量', '平均单价']]


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    按国家汇总业务表现。

    复购率 = 该国内发票数 > 1 的客户占该国客户数的比例, 用来区分"一次性买家聚集地"
    与"稳定复购市场"——单看收入会漏掉这个差别。

    Parameters:
        df: 清洗后交易数据

    Returns:
        DataFrame: 国家, 客户数, 收入, 收入占比%, 订单数, 客单价, 复购率%, 商品种类数
    """
    base = df.groupby('Country').agg(
        客户数=('Customer ID', 'nunique'),
        收入=('Revenue', 'sum'),
        订单数=('Invoice', 'nunique'),
        商品种类数=('StockCode', 'nunique'),
    ).reset_index()
    base['客单价'] = (base['收入'] / base['订单数']).round(2)
    base['收入占比%'] = (base['收入'] / base['收入'].sum() * 100).round(2)

    per_cust = df.groupby(['Country', 'Customer ID'])['Invoice'].nunique()
    repeat_rate = per_cust.gt(1).groupby(level=0).mean() * 100
    base['复购率%'] = repeat_rate.reindex(base['Country'].values).round(1).values

    base['收入'] = base['收入'].round(0)
    return base.sort_values('收入', ascending=False).reset_index(drop=True)


def uk_vs_overseas_profile(df: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    本土 (英国) vs 海外 的客户画像对比。

    Country 直接进 K-Means 几乎没有区分度 (91% 是英国), 但把客户二分成本土/海外后,
    两组的 RFM 与扩展特征差异是可解读的 —— 这是 Country 这个特征最合适的用法。

    Parameters:
        df: 清洗后交易数据 (提供每个客户所属国家)
        rfm_df: 客户级数据, 需含 Recency/Frequency/Monetary, 可选含 AOV / N_products

    Returns:
        DataFrame, 每行一个区域: 客户数, 各特征均值, 收入占比%
    """
    cust_country = df.groupby('Customer ID')['Country'].first()
    prof = rfm_df.set_index('Customer ID').copy()
    prof['区域'] = np.where(
        cust_country.reindex(prof.index).eq('United Kingdom'), '英国本土', '海外')

    cols = ['Recency', 'Frequency', 'Monetary']
    cols += [c for c in ['AOV', 'N_products'] if c in prof.columns]

    out = prof.groupby('区域')[cols].mean().round(2)
    out.insert(0, '客户数', prof.groupby('区域').size())
    out['收入占比%'] = (prof.groupby('区域')['Monetary'].sum()
                       / prof['Monetary'].sum() * 100).round(1)
    return out


def country_monthly_revenue(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    收入 Top-N 国家的月度收入透视表 (供堆叠面积图观察海外市场增长)。

    Parameters:
        df: 清洗后交易数据
        top_n: 取收入最高的前 N 个国家

    Returns:
        DataFrame: 月份 + 每个国家一列的收入
    """
    top = df.groupby('Country')['Revenue'].sum().nlargest(top_n).index.tolist()
    sub = df[df['Country'].isin(top)].copy()
    sub['月份'] = sub['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
    pivot = sub.pivot_table(index='月份', columns='Country', values='Revenue',
                            aggfunc='sum').fillna(0)
    return pivot.reset_index()
