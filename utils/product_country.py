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
        DataFrame: 关键词, 商品数, 收入, 收入占比%, 销量, 平均单价
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
    return out.sort_values('收入', ascending=False).head(top_n).reset_index(drop=True)


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
