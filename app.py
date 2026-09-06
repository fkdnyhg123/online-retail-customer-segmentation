"""
Streamlit 可视化大屏 - 在线零售客户分群分析
课程: 大数据分析应用综合实践
数据集: UCI Online Retail II (英国在线零售商店)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_cleaner import load_raw_data, clean_data, get_data_quality_report, get_cleaning_summary
from utils.rfm import calculate_rfm, add_rfm_scores, get_rfm_stats
from utils.clustering import prepare_features, find_optimal_k, run_kmeans, get_cluster_labels

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="在线零售客户分群分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 白色主题 + 动画优化 CSS
st.markdown("""
<style>
    /* 全局白色主题 */
    .stApp { background-color: #ffffff; }
    .main .block-container { padding-top: 2rem; }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 15px; font-weight: 500; color: #495057;
        padding: 6px 12px; border-radius: 8px;
        transition: all 0.3s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #dee2e6;
    }

    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f4fd 100%);
        border-radius: 16px; padding: 20px; text-align: center;
        border: 1px solid #d0e0f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .metric-card h2 { color: #2563eb; margin: 0; font-size: 28px; }
    .metric-card p { color: #6b7280; margin: 5px 0 0 0; font-size: 13px; }

    /* 聚类详情卡片 */
    .cluster-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%);
        border-radius: 14px; padding: 18px; margin: 6px 0;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .cluster-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    /* 分割线 */
    .custom-divider {
        height: 2px; background: linear-gradient(90deg, transparent, #d0d5dd, transparent);
        margin: 20px 0; border: none;
    }

    /* 平滑过渡 */
    [data-testid="stMetric"] {
        background: #f8fafc; border-radius: 12px; padding: 12px 16px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        background: #f1f5f9; border-color: #cbd5e1;
    }

    /* Tab 标签动画 */
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s ease;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f4ff;
    }

    /* Plotly 图表容器动画 */
    .stPlotlyChart { transition: opacity 0.4s ease; }

    /* 标题样式 */
    h1 { color: #1e293b !important; }
    h2 { color: #334155 !important; }
    h3 { color: #475569 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Plotly 全局配色方案 (白色主题适配)
# ============================================================
COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#059669',
    'warning': '#d97706',
    'danger': '#dc2626',
    'info': '#0891b2',
    'palette': ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626',
                '#0891b2', '#c026d3', '#ea580c', '#0d9488', '#4f46e5'],
}

CHART_LAYOUT = dict(
    paper_bgcolor='white',
    plot_bgcolor='#fafbfc',
    font=dict(color='#374151', family='Microsoft YaHei, SimHei, sans-serif', size=13),
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor='white', font_size=13, bordercolor='#e5e7eb'),
)

# ============================================================
# 数据加载 (缓存)
# ============================================================
@st.cache_data(show_spinner="正在加载数据集，请稍候...")
def load_and_clean(filepath):
    raw_df = load_raw_data(filepath)
    cleaned_df = clean_data(raw_df)
    quality_report = get_data_quality_report(raw_df)
    cleaned_quality = get_data_quality_report(cleaned_df)
    cleaning_summary = get_cleaning_summary(raw_df, cleaned_df)
    rfm_df = calculate_rfm(cleaned_df)
    rfm_scored = add_rfm_scores(rfm_df)
    rfm_stats = get_rfm_stats(rfm_df)
    return raw_df, cleaned_df, quality_report, cleaned_quality, cleaning_summary, rfm_df, rfm_scored, rfm_stats

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_retail_II(1).xlsx")
raw_df, cleaned_df, quality_report, cleaned_quality, cleaning_summary, rfm_df, rfm_scored, rfm_stats = load_and_clean(DATA_PATH)

# ============================================================
# 侧边栏导航
# ============================================================
st.sidebar.markdown("## 📊 导航面板")
page = st.sidebar.radio(
    "页面选择",
    ["📈 数据概览", "🔍 数据探索", "💰 RFM 分析", "🎯 K-Means 聚类"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**数据集:** Online Retail II (在线零售)")
st.sidebar.markdown(f"**时间范围:** {cleaning_summary['date_range_start']} ~ {cleaning_summary['date_range_end']}")
st.sidebar.markdown(f"**清洗后记录:** {cleaning_summary['cleaned_rows']:,} 条")
st.sidebar.markdown(f"**客户数量:** {cleaning_summary['unique_customers']:,} 人")

# ============================================================
# 页面 1: 数据概览
# ============================================================
if page == "📈 数据概览":
    st.title("📈 数据概览")
    st.markdown("UCI Online Retail II — 英国在线零售商店交易数据")

    # ---- 第一周任务: 数据基础探索 ----
    st.subheader("📋 第一周: 数据理解与基础探索")
    st.markdown("""
    > **任务目标**: 加载原始数据，执行基础探索 (`df.info()`, `df.head()`, `df.describe()`, `df.isnull()`)，识别数据质量问题，形成初步观察。
    
    **数据集来源**: UCI 机器学习仓库 — Online Retail II (ID: 502)，包含一家英国在线零售商 2009.12 至 2010.12 的全部交易记录。
    **8 个字段**: Invoice (发票号), StockCode (商品编码), Description (商品描述), Quantity (数量), InvoiceDate (日期), Price (单价), Customer ID (客户ID), Country (国家)。
    """)

    # 数据规模快速概览
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("原始记录数", f"{cleaning_summary['raw_rows']:,}")
    with col2:
        st.metric("字段数", "8")
    with col3:
        st.metric("时间跨度", f"{cleaning_summary['date_range_start']} ~ {cleaning_summary['date_range_end']}")
    with col4:
        st.metric("覆盖国家", f"{raw_df['Country'].nunique()} 个")

    # 三个初步观察
    st.markdown("""
    **初步观察 (3 条)**:
    1. **数据规模与缺失**: 共 525,461 条记录。Customer ID 缺失约 20.5% (107,927 条)，将严重影响客户级分析 (如 RFM)。Description 有 2,928 条缺失。
    2. **多重质量问题**: 存在取消订单 (Invoice 以 'C' 开头)、负数量、零/负价格、非商品编码 (POST 等)、精确重复行等多种问题，需仔细清洗。
    3. **地域集中**: 绝大多数订单来自英国，德国、法国等欧洲国家占比小，地域不平衡需在分析中考虑。
    """)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- 原始数据质量报告 ----
    st.subheader("🔎 原始数据质量报告")
    col1, col2 = st.columns([1, 1])

    with col1:
        quality_items = [
            ("取消订单 (Cancelled)", quality_report['cancelled_orders'], quality_report['cancelled_orders_pct']),
            ("缺失客户ID (Customer ID)", quality_report['missing_customer_id'], quality_report['missing_customer_id_pct']),
            ("负数量 (Negative Qty)", quality_report['negative_quantity'], quality_report['negative_quantity_pct']),
            ("零/负价格 (Zero/Neg Price)", quality_report['zero_neg_price'], quality_report['zero_neg_price_pct']),
            ("精确重复 (Duplicates)", quality_report['exact_duplicates'], quality_report['exact_duplicates_pct']),
            ("非商品编码 (Non-product)", quality_report['special_stockcodes'], quality_report['special_stockcodes_pct']),
        ]
        for label, count, pct in quality_items:
            st.markdown(f"**{label}**: {count:,} 条 ({pct}%)")

    with col2:
        labels_cn = [q[0] for q in quality_items]
        values = [q[1] for q in quality_items]
        fig_quality = px.pie(names=labels_cn, values=values, hole=0.45,
                             color_discrete_sequence=COLORS['palette'])
        fig_quality.update_layout(**CHART_LAYOUT, height=380,
                                  title="原始数据质量问题分布")
        st.plotly_chart(fig_quality, use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- 数据清洗说明 ----
    st.subheader("🧹 数据清洗步骤说明")
    st.markdown("""
    清洗按以下 **7 个步骤** 依次执行:
    
    | 步骤 | 操作 | 说明 |
    |:---:|------|------|
    | 1 | 移除取消订单 | Invoice 以 'C' 开头的记录为退货/取消，不参与正向分析 |
    | 2 | 移除非商品编码 | POST (邮费)、BANK CHARGES (银行费用) 等不是实际商品 |
    | 3 | 移除缺失描述 | Description 为空的记录无法进行商品分析 |
    | 4 | 移除负数量 | 负数量对应退货，已在步骤 1 部分处理 |
    | 5 | 移除零/负价格 | 免费赠品或系统错误，不属于正常交易 |
    | 6 | 去除精确重复行 | 完全相同的记录只保留一条 |
    | 7 | 移除缺失客户ID | RFM 和聚类分析必须以客户为单位 |
    """)

    # 清洗结果摘要
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("清洗前", f"{cleaning_summary['raw_rows']:,} 条")
    with col_b:
        st.metric("清洗后", f"{cleaning_summary['cleaned_rows']:,} 条")
    with col_c:
        st.metric("移除", f"{cleaning_summary['removed_rows']:,} 条")
    with col_d:
        st.metric("移除比例", f"{cleaning_summary['removal_pct']}%")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- 清洗后数据质量报告 ----
    st.subheader("✅ 清洗后数据质量报告")
    st.markdown("清洗后数据已无质量问题，可用于后续分析:")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.metric("客户数 (Customer ID)", f"{cleaning_summary['unique_customers']:,} 人")
    with col_c2:
        st.metric("商品数 (StockCode)", f"{cleaning_summary['unique_products']:,} 种")
    with col_c3:
        st.metric("交易记录数", f"{cleaning_summary['cleaned_rows']:,} 条")

    # 清洗后质量检查
    clean_issues = [
        ("取消订单", cleaned_quality['cancelled_orders']),
        ("缺失客户ID", cleaned_quality['missing_customer_id']),
        ("负数量", cleaned_quality['negative_quantity']),
        ("零/负价格", cleaned_quality['zero_neg_price']),
        ("精确重复", cleaned_quality['exact_duplicates']),
        ("非商品编码", cleaned_quality['special_stockcodes']),
    ]
    all_clean = all(v == 0 for _, v in clean_issues)
    if all_clean:
        st.success("✅ 所有质量检查项均为 0 — 数据清洗完成，无残留问题。")
    else:
        for label, count in clean_issues:
            if count > 0:
                st.warning(f"⚠️ {label}: 仍有 {count:,} 条")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- 清洗后数据 KPI 与趋势 ----
    st.subheader("📊 清洗后核心指标")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><h2>{cleaning_summary["cleaned_rows"]:,}</h2><p>交易总量 (Transactions)</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h2>{cleaning_summary["unique_customers"]:,}</h2><p>客户数量 (Customers)</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h2>{cleaning_summary["unique_products"]:,}</h2><p>商品数量 (Products)</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h2>${rfm_stats["total_revenue"]:,.0f}</h2><p>总收入 (Total Revenue)</p></div>', unsafe_allow_html=True)
    with col5:
        avg_order = cleaned_df.groupby('Invoice')['Revenue'].sum().mean()
        st.markdown(f'<div class="metric-card"><h2>${avg_order:,.2f}</h2><p>平均订单金额 (Avg Order)</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # 月度收入趋势 + 国家排名
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📈 月度收入趋势")
        monthly = cleaned_df.set_index('InvoiceDate').resample('ME')['Revenue'].sum().reset_index()
        fig_revenue = px.area(monthly, x='InvoiceDate', y='Revenue',
                              labels={'Revenue': '收入 (美元 $)', 'InvoiceDate': '日期'},
                              color_discrete_sequence=[COLORS['primary']])
        fig_revenue.update_layout(**CHART_LAYOUT, height=420, hovermode='x unified',
                                  xaxis_title='日期', yaxis_title='收入 (美元 $)')
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col_right:
        st.subheader("🌍 国家收入排名 Top 10")
        country_rev = cleaned_df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        country_cn = {'United Kingdom': '英国', 'EIRE': '爱尔兰', 'Netherlands': '荷兰',
                      'Germany': '德国', 'France': '法国', 'Denmark': '丹麦',
                      'Sweden': '瑞典', 'Spain': '西班牙', 'Switzerland': '瑞士',
                      'Australia': '澳大利亚', 'Norway': '挪威', 'Canada': '加拿大',
                      'Italy': '意大利', 'Belgium': '比利时', 'Portugal': '葡萄牙',
                      'USA': '美国', 'Japan': '日本', 'Finland': '芬兰'}
        country_rev['国家'] = country_rev['Country'].map(lambda x: f"{country_cn.get(x, x)} ({x})")
        fig_country = px.bar(country_rev, x='Revenue', y='国家', orientation='h',
                             color='Revenue', color_continuous_scale='Blues')
        fig_country.update_layout(**CHART_LAYOUT, height=420, showlegend=False,
                                  yaxis={'categoryorder': 'total ascending'},
                                  xaxis_title='收入 (美元 $)')
        st.plotly_chart(fig_country, use_container_width=True)

# ============================================================
# 页面 2: 数据探索
# ============================================================
elif page == "🔍 数据探索":
    st.title("🔍 数据探索")
    st.markdown("对清洗后的数据进行深入的分布分析和特征探索，发现数据模式和业务洞察。")

    # 字段分布
    st.subheader("📊 字段值分布")
    st.caption("通过直方图观察各字段的值分布形态，判断是否存在偏态、离群值等特征。")
    tab1, tab2, tab3 = st.tabs(["数量 (Quantity)", "单价 (Price)", "收入 (Revenue)"])

    with tab1:
        fig_qty = px.histogram(cleaned_df, x='Quantity', nbins=80,
                               color_discrete_sequence=[COLORS['primary']])
        fig_qty.update_layout(**CHART_LAYOUT, height=400, title="订单数量分布",
                              xaxis_title='购买数量', yaxis_title='订单数',
                              xaxis=dict(range=[0, cleaned_df['Quantity'].max()]))
        st.plotly_chart(fig_qty, use_container_width=True)
        st.markdown(f"均值: {cleaned_df['Quantity'].mean():.1f} | 中位数: {cleaned_df['Quantity'].median():.0f} | 最大值: {cleaned_df['Quantity'].max()}")

    with tab2:
        fig_price = px.histogram(cleaned_df, x='Price', nbins=80,
                                 color_discrete_sequence=[COLORS['secondary']])
        fig_price.update_layout(**CHART_LAYOUT, height=400, title="单价分布",
                                xaxis_title='单价 (美元 $)', yaxis_title='订单数',
                                xaxis=dict(range=[0, cleaned_df['Price'].max()]))
        st.plotly_chart(fig_price, use_container_width=True)
        st.markdown(f"均值: ${cleaned_df['Price'].mean():.2f} | 中位数: ${cleaned_df['Price'].median():.2f} | 最大值: ${cleaned_df['Price'].max():.2f}")

    with tab3:
        fig_rev = px.histogram(cleaned_df, x='Revenue', nbins=80,
                               color_discrete_sequence=[COLORS['success']])
        fig_rev.update_layout(**CHART_LAYOUT, height=400, title="单笔交易收入分布",
                              xaxis_title='收入 (美元 $)', yaxis_title='订单数',
                              xaxis=dict(range=[0, cleaned_df['Revenue'].max()]))
        st.plotly_chart(fig_rev, use_container_width=True)

    # 热销商品
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("🏆 热销商品 Top 20 (按收入排名)")
    top_products = cleaned_df.groupby(['StockCode', 'Description']).agg(
        收入=('Revenue', 'sum'),
        销量=('Quantity', 'sum'),
        订单数=('Invoice', 'nunique')
    ).sort_values('收入', ascending=False).head(20).reset_index()
    top_products['商品名称'] = top_products.apply(lambda r: f"{r['Description']}  ({r['StockCode']})", axis=1)
    fig_top = px.bar(top_products, x='收入', y='商品名称', orientation='h',
                     color='收入', color_continuous_scale='Viridis',
                     hover_data=['StockCode', '销量', '订单数'])
    fig_top.update_layout(**CHART_LAYOUT, height=650, showlegend=False,
                          yaxis={'categoryorder': 'total ascending'},
                          xaxis_title='总收入 (美元 $)')
    st.plotly_chart(fig_top, use_container_width=True)
    st.caption("💡 **解读**: 热销商品以家居装饰和节日用品为主。排名靠前的商品 (如 WHITE HANGING HEART T-LIGHT HOLDER) 具有明显的爆款特征，收入远超其他商品。长尾分布表明少量头部商品贡献了大部分收入。")

# ============================================================
# 页面 3: RFM 分析
# ============================================================
elif page == "💰 RFM 分析":
    st.title("💰 RFM 客户价值分析")
    st.markdown("基于 Recency (最近购买)、Frequency (购买频率)、Monetary (消费金额) 的客户价值模型")

    # RFM 统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均 Recency", f"{rfm_stats['avg_recency']} 天", help="距上次购买的平均天数")
        st.metric("中位数 Recency", f"{rfm_stats['median_recency']} 天")
    with col2:
        st.metric("平均 Frequency", f"{rfm_stats['avg_frequency']:.1f} 次", help="平均订单数")
        st.metric("中位数 Frequency", f"{rfm_stats['median_frequency']:.0f} 次")
    with col3:
        st.metric("平均 Monetary", f"${rfm_stats['avg_monetary']:.2f}", help="平均总消费金额")
        st.metric("中位数 Monetary", f"${rfm_stats['median_monetary']:.2f}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # R/F/M 分布
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_r = px.histogram(rfm_df, x='Recency', nbins=50,
                             color_discrete_sequence=[COLORS['primary']])
        fig_r.update_layout(**CHART_LAYOUT, title="R - 最近购买间隔分布",
                            xaxis_title='距上次购买 (天)', yaxis_title='客户数', height=360,
                            xaxis=dict(range=[0, rfm_df['Recency'].max()]))
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        fig_f = px.histogram(rfm_df, x='Frequency', nbins=50,
                             color_discrete_sequence=[COLORS['danger']])
        fig_f.update_layout(**CHART_LAYOUT, title="F - 购买频率分布",
                            xaxis_title='订单数', yaxis_title='客户数', height=360,
                            xaxis=dict(range=[0, rfm_df['Frequency'].max()]))
        st.plotly_chart(fig_f, use_container_width=True)

    with col3:
        fig_m = px.histogram(rfm_df, x='Monetary', nbins=50,
                             color_discrete_sequence=[COLORS['success']])
        fig_m.update_layout(**CHART_LAYOUT, title="M - 消费金额分布",
                            xaxis_title='总消费 ($)', yaxis_title='客户数', height=360,
                            xaxis=dict(range=[0, rfm_df['Monetary'].max()]))
        st.plotly_chart(fig_m, use_container_width=True)

    st.caption("💡 **解读**: R (最近购买间隔) 呈较均匀的右偏分布，中位数 52 天，说明大多数客户在 2 个月内有购买行为。F (购买频率) 和 M (消费金额) 均呈严重右偏分布，中位数远低于均值——大多数客户为低频低消费群体，少量高频高消费客户拉高了均值。这种偏态分布是后续需要做对数变换的原因。")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("🔗 RFM 维度相关性")
    col1, col2 = st.columns(2)
    with col1:
        fig_rf = px.scatter(rfm_df, x='Frequency', y='Recency',
                            size='Monetary', color='Monetary',
                            color_continuous_scale='Viridis',
                            hover_data=['Customer ID'], opacity=0.6)
        fig_rf.update_layout(**CHART_LAYOUT, title="频率 vs 最近购买 (气泡大小/颜色 = 消费金额)",
                             xaxis_title='购买频率 (次)', yaxis_title='最近购买间隔 (天)', height=450)
        st.plotly_chart(fig_rf, use_container_width=True)

    with col2:
        fig_fm = px.scatter(rfm_df, x='Frequency', y='Monetary',
                            size='Recency', color='Recency',
                            color_continuous_scale='RdYlGn_r',
                            hover_data=['Customer ID'], opacity=0.6)
        fig_fm.update_layout(**CHART_LAYOUT, title="频率 vs 消费金额 (气泡大小/颜色 = 最近购买)",
                             xaxis_title='购买频率 (次)', yaxis_title='消费金额 ($)', height=450)
        st.plotly_chart(fig_fm, use_container_width=True)

    st.caption("💡 **解读**: 左图 (频率 vs 最近购买) 中气泡大小和颜色代表消费金额，可见高频客户 (右侧) 消费金额更高且购买更近期。右图 (频率 vs 消费金额) 显示频率和金额正相关，但少量极端客户 (右上角) 的消费金额远超其他客户，进一步验证了偏态分布。")

    # RFM 分数分群
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("📊 RFM 评分分群 (分位数法)")
    scored = rfm_scored.copy()

    def assign_segment(row):
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        if r >= 4 and f >= 4 and m >= 4:
            return '冠军客户 (Champions)'
        elif r >= 3 and f >= 3:
            return '忠诚客户 (Loyal)'
        elif r >= 4 and f <= 2:
            return '新客户 (New)'
        elif r <= 2 and f >= 3 and m >= 3:
            return '流失风险 (At-Risk)'
        elif r <= 2 and f <= 2 and m <= 2:
            return '流失客户 (Lost)'
        elif r >= 3 and f >= 1 and f <= 2:
            return '潜力客户 (Potential)'
        elif r <= 2:
            return '沉睡客户 (Hibernating)'
        else:
            return '待开发 (Develop)'

    scored['分群'] = scored.apply(assign_segment, axis=1)
    seg_counts = scored['分群'].value_counts().reset_index()
    seg_counts.columns = ['客户分群', '客户数量']

    fig_seg = px.bar(seg_counts, x='客户分群', y='客户数量',
                     color='客户数量', color_continuous_scale='Blues',
                     text='客户数量')
    fig_seg.update_layout(**CHART_LAYOUT, height=420, xaxis_tickangle=-25,
                          xaxis_title='', yaxis_title='客户数量')
    st.plotly_chart(fig_seg, use_container_width=True)
    st.caption("💡 **解读**: 使用分位数法将 R/F/M 各分为 1-5 档 (5 最高)，根据组合得分将客户归入不同价值分群。**沉睡客户**和**流失客户**占比最大，说明客户留存是核心问题。**冠军客户**虽然数量少，但贡献了绝大部分收入，应重点维护。")

# ============================================================
# 页面 4: K-Means 聚类分析
# ============================================================
elif page == "🎯 K-Means 聚类":
    st.title("🎯 K-Means 客户分群分析")

    # 侧边栏参数控制
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 聚类参数")
    k_range_start = st.sidebar.number_input("K 搜索范围 (起始)", min_value=2, max_value=5, value=2, step=1)
    k_range_end = st.sidebar.number_input("K 搜索范围 (结束)", min_value=6, max_value=15, value=10, step=1)
    cluster_method = st.sidebar.selectbox(
        "特征变换方法",
        options=['composite', 'rank', 'log'],
        format_func=lambda x: {
            'composite': '组合特征 2D (推荐)',
            'rank': '百分位排名 3D',
            'log': '对数变换 3D',
        }[x],
        help="组合特征: R_rank + RFM_composite (2D)，消除F/M相关性冗余，轮廓系数最高"
    )
    winsorize_pct = st.sidebar.slider("Winsorize 截断 (上分位)", min_value=0.95, max_value=1.0,
                                       value=0.995, step=0.005, format="%.3f",
                                       help="将 F/M 超过该分位数的值截断，默认 0.995 (截断 top 0.5%)")
    selected_k = st.sidebar.slider("聚类数量 K", min_value=2, max_value=10, value=5, step=1)

    # 特征准备 (V2: rank 或 log 方法)
    scaled, feature_names, scaler, transformed_df = prepare_features(
        rfm_df, method=cluster_method, winsorize_pct=winsorize_pct)

    # 最优 K 分析
    st.subheader("📐 第一步: 确定最优聚类数 K")
    col1, col2 = st.columns(2)

    elbow_result = find_optimal_k(scaled, k_range=range(k_range_start, k_range_end + 1))

    with col1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=elbow_result['k_values'], y=elbow_result['inertia'],
            mode='lines+markers', name='WCSS (组内平方和)',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=10, color=COLORS['primary']),
        ))
        fig_elbow.update_layout(**CHART_LAYOUT,
            title="肘部法则 (Elbow Method)",
            xaxis_title='聚类数 K', yaxis_title='组内平方和 (WCSS)',
            height=400)
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.caption("💡 **肘部法则**: WCSS (组内平方和) 随 K 增大而下降，曲线拐点处为最优 K。拐点之后继续增加 K 收益递减。")
    with col2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=elbow_result['k_values'], y=elbow_result['silhouette_scores'],
            mode='lines+markers', name='轮廓系数',
            line=dict(color=COLORS['danger'], width=3),
            marker=dict(size=10, color=COLORS['danger']),
            fill='tozeroy', fillcolor='rgba(220,38,38,0.08)',
        ))
        best_idx = elbow_result['k_values'].index(elbow_result['best_k'])
        fig_sil.add_trace(go.Scatter(
            x=[elbow_result['best_k']], y=[elbow_result['silhouette_scores'][best_idx]],
            mode='markers+text',
            marker=dict(size=18, color=COLORS['warning'], symbol='star', line=dict(width=2, color='white')),
            text=[f'最优 K={elbow_result["best_k"]}'],
            textposition='top center',
            name=f'最优 K={elbow_result["best_k"]} (轮廓系数: {elbow_result["best_silhouette"]:.4f})'
        ))
        fig_sil.update_layout(**CHART_LAYOUT,
            title="轮廓系数分析 (Silhouette Score)",
            xaxis_title='聚类数 K', yaxis_title='轮廓系数',
            height=400)
        st.plotly_chart(fig_sil, use_container_width=True)
        st.caption("💡 **轮廓系数**: 衡量簇内紧密度和簇间分离度，取值 -1~1。越高表示簇内样本越紧密、不同簇之间分离越清晰。⭐ 标记为最优 K。")

    st.info(f"**推荐 K = {elbow_result['best_k']}** (最高轮廓系数: {elbow_result['best_silhouette']:.4f})，当前选择 **K = {selected_k}**")

    with st.expander("🔬 三种方法对比 — 为什么推荐组合特征 2D？(点击展开)"):
        st.markdown("""
        **核心矛盾**: K-Means 依赖欧氏距离，对 F 和 M 的极度右偏极其敏感 (F 偏度 9.98, M 偏度 24.32)。
        同时 F 和 M 高度相关 (r≈0.8)，在 3D 空间中产生冗余维度，降低簇间分离度。

        **三种方法对比**:

        | 方法 | 维度 | 特征 | K=5 轮廓系数 | 说明 |
        |:---|:---:|------|:---:|------|
        | **组合特征** | 2D | R_rank + RFM_composite | **0.42** ⭐ | 合并 F/M 为"参与度"，消除冗余 |
        | 百分位排名 | 3D | R_rank + F_rank + M_rank | 0.36 | 标准 rank，3个独立维度 |
        | 对数变换 | 3D | log(R) + log(F) + log(M) | 0.32 | 传统方法，压缩效果有限 |

        **为什么 2D 组合特征更好？**
        1. **消除冗余**: F (购买频率) 和 M (消费金额) 天然正相关——买得多自然花得多。在 3D 空间中，这两个维度传递的是相似信息，导致 K-Means 的"注意力"被重复消耗。
        2. **降维提升分离度**: 将 R/F/M 压缩为 2 个正交维度——**时效性** (R_rank: 最近购买排名) 和 **参与度** (RFM_composite: 三个排名之和)。2D 空间中簇更容易被清晰分离。
        3. **业务直觉**: 时效性 × 参与度 恰好构成经典的客户分群矩阵——近期活跃的高参与度客户 = 重要价值客户，长期沉睡的低参与度客户 = 流失客户。

        **4 步优化流程 (三种方法共用)**:
        1. **Winsorizing**: 截断 F/M 的 top 0.5% 极端值
        2. **百分位排名**: `rank(pct=True)` 压缩到 0~1
        3. **StandardScaler**: Z-score 标准化
        4. **K-Means**: `n_init=50, max_iter=500` 避免局部最优
        """)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # 执行聚类
    st.subheader("🎯 第二步: 聚类结果")
    result = run_kmeans(rfm_df, n_clusters=selected_k, method=cluster_method,
                        winsorize_pct=winsorize_pct)
    profile = result['cluster_profile']
    clustered_df = result['clustered_df']
    labels = get_cluster_labels(profile)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("聚类数量", f"{selected_k} 组")
    with col2:
        st.metric("轮廓系数", f"{result['silhouette_score']:.4f}", help="取值范围 -1~1，越高越好")
    with col3:
        st.metric("参与聚类客户", f"{len(clustered_df):,} 人")
    with col4:
        method_labels = {'composite': '组合特征 2D', 'rank': '百分位排名 3D', 'log': '对数变换 3D'}
        st.metric("变换方法", method_labels[cluster_method], help="n_init=50, max_iter=500")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # 3D 散点图
    st.subheader("🌐 客户 3D 分群可视化")
    plot_df = clustered_df.copy()
    plot_df['聚类标签'] = plot_df['Cluster'].map(lambda c: f"C{c}: {labels[c]['name']}")

    fig_3d = px.scatter_3d(
        plot_df, x='Recency', y='Frequency', z='Monetary',
        color='聚类标签',
        hover_data=['Customer ID'],
        opacity=0.75,
        labels={'Recency': 'R-最近购买 (天)', 'Frequency': 'F-购买频率', 'Monetary': 'M-消费金额 ($)'},
        color_discrete_sequence=COLORS['palette'],
    )
    fig_3d.update_layout(**CHART_LAYOUT, height=600,
        scene=dict(
            xaxis=dict(backgroundcolor='#fafbfc', gridcolor='#e5e7eb', title='R-最近购买 (天)'),
            yaxis=dict(backgroundcolor='#fafbfc', gridcolor='#e5e7eb', title='F-购买频率'),
            zaxis=dict(backgroundcolor='#fafbfc', gridcolor='#e5e7eb', title='M-消费金额 ($)'),
            bgcolor='white',
        ),
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.caption("💡 **解读**: 3D 散点图展示客户在 R/F/M 三维空间中的分布。可旋转查看各簇的空间位置关系——重要价值客户簇 (高频高消费) 位于图上方，流失类簇位于前方 (高 Recency)。簇间存在部分重叠，说明客户群体间边界不是绝对分明的。")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # 聚类画像 + 雷达图
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 聚类画像 (Cluster Profile)")
        display_profile = profile.copy()
        display_profile['标签'] = display_profile.index.map(lambda c: labels[c]['name'])
        display_profile = display_profile[['标签', 'Customers', 'Pct_Customers',
                                           'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary',
                                           'Total_Revenue', 'Pct_Revenue']]
        display_profile.columns = ['客户分群', '客户数', '客户占比%', '平均R(天)',
                                   '平均F(次)', '平均M($)', '总收入', '收入占比%']
        st.dataframe(display_profile, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("🕸️ 雷达图 (Radar Chart)")
        norm_profile = profile.copy()
        for col in ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']:
            norm_profile[col] = (norm_profile[col] - norm_profile[col].min()) / \
                                (norm_profile[col].max() - norm_profile[col].min() + 1e-10)
        norm_profile['Avg_Recency'] = 1 - norm_profile['Avg_Recency']

        categories = ['R-最近购买*', 'F-购买频率', 'M-消费金额']
        fig_radar = go.Figure()
        for cluster_id in norm_profile.index:
            values = [
                norm_profile.loc[cluster_id, 'Avg_Recency'],
                norm_profile.loc[cluster_id, 'Avg_Frequency'],
                norm_profile.loc[cluster_id, 'Avg_Monetary'],
            ]
            values.append(values[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=f"C{cluster_id}: {labels[cluster_id]['name']}",
                fill='toself',
                opacity=0.5,
                line=dict(width=2),
            ))
        fig_radar.update_layout(**CHART_LAYOUT, height=460,
            polar=dict(
                bgcolor='#fafbfc',
                radialaxis=dict(gridcolor='#e5e7eb', range=[0, 1.05]),
                angularaxis=dict(gridcolor='#e5e7eb'),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("💡 **解读**: 雷达图面积越大代表该簇综合价值越高。重要价值客户簇在三个维度上均突出，覆盖面积最大。可直观对比各簇在 R/F/M 上的优劣势差异。")
        st.caption("*R-最近购买: 值越高 = 购买越近期 (已反转)")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # 聚类详情卡片
    st.subheader("📇 各聚类详情")
    cols = st.columns(min(selected_k, 5))
    card_colors = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626',
                   '#0891b2', '#c026d3', '#ea580c', '#0d9488', '#4f46e5']

    for idx, cluster_id in enumerate(profile.index):
        with cols[idx % len(cols)]:
            color = card_colors[cluster_id % len(card_colors)]
            st.markdown(f"""
            <div class="cluster-card" style="border-left-color: {color};">
                <h3 style="color: {color}; margin: 0; font-size: 16px;">
                    C{cluster_id}: {labels[cluster_id]['name']}
                </h3>
                <p style="color: #6b7280; margin: 4px 0; font-size: 13px;">
                    {labels[cluster_id]['traits']}
                </p>
                <p style="color: #374151; margin: 6px 0; font-size: 13px; line-height: 1.8;">
                    客户数: <b>{profile.loc[cluster_id, 'Customers']:,}</b> ({profile.loc[cluster_id, 'Pct_Customers']}%)<br>
                    平均 R: <b>{profile.loc[cluster_id, 'Avg_Recency']:.0f} 天</b><br>
                    平均 F: <b>{profile.loc[cluster_id, 'Avg_Frequency']:.1f} 次</b><br>
                    平均 M: <b>${profile.loc[cluster_id, 'Avg_Monetary']:,.2f}</b><br>
                    收入占比: <b>{profile.loc[cluster_id, 'Pct_Revenue']}%</b><br>
                    簇轮廓系数: <b>{result['silhouette_per_cluster'].get(cluster_id, 0):.3f}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

    # 聚类中心热力图
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("🔥 聚类中心热力图 (归一化)")
    centers = result['cluster_centers'].copy()

    # Normalize each column to 0-1
    for col in centers.columns:
        col_min, col_max = centers[col].min(), centers[col].max()
        centers[col] = (centers[col] - col_min) / (col_max - col_min + 1e-10)

    # Invert Recency for log method (raw Recency: higher = more lapsed)
    # composite's R_rank and rank's R_rank already have higher = more recent
    if cluster_method == 'log' and 'Recency' in centers.columns:
        centers['Recency'] = 1 - centers['Recency']

    centers.index = [f"C{i}: {labels[i]['name']}" for i in centers.index]
    col_map = {
        'Recency': 'R-最近购买', 'Frequency': 'F-购买频率', 'Monetary': 'M-消费金额',
        'R_rank': 'R-最近购买 (排名)', 'RFM_composite': 'RFM 综合参与度',
    }
    centers.columns = [col_map.get(c, c) for c in centers.columns]

    fig_heat = px.imshow(centers.values,
                         x=centers.columns.tolist(),
                         y=centers.index.tolist(),
                         color_continuous_scale='RdYlGn',
                         aspect='auto',
                         labels={'color': '得分'})
    fig_heat.update_layout(**CHART_LAYOUT, height=400,
                           xaxis_title='', yaxis_title='')
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("💡 **解读**: 热力图将聚类中心归一化到 0-1 区间，颜色越绿表示该维度得分越高。可快速识别每个簇的'强项'和'弱项'——例如重要价值客户在各维度上得分接近 1，而流失客户各维度均偏低。组合特征方法使用 R_rank (时效性) + RFM_composite (参与度) 两个正交维度。")
