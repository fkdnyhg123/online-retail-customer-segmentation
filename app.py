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

    /* 日期范围卡片 (更宽) */
    .date-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f4fd 100%);
        border-radius: 16px; padding: 16px 20px; text-align: center;
        border: 1px solid #d0e0f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .date-card h3 { color: #2563eb; margin: 0; font-size: 20px; letter-spacing: 1px; }
    .date-card p { color: #6b7280; margin: 4px 0 0 0; font-size: 13px; }

    /* 带左侧色条的标题 */
    .section-header {
        border-left: 4px solid #2563eb;
        padding-left: 12px;
        margin: 24px 0 12px 0;
    }

    /* 信息提示条 */
    .info-banner {
        background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
        border-radius: 12px; padding: 14px 20px;
        border: 1px solid #bfdbfe;
        margin: 12px 0;
        font-size: 13px; color: #374151;
    }

    /* 宽幅指标卡 */
    .metric-card-wide {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f4fd 100%);
        border-radius: 16px; padding: 18px 28px; text-align: center;
        border: 1px solid #d0e0f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card-wide:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .metric-card-wide h2 { color: #2563eb; margin: 0; font-size: 24px; }
    .metric-card-wide p { color: #6b7280; margin: 5px 0 0 0; font-size: 13px; }

    /* 紧凑等高统计卡片行 */
    .stat-row { display: flex; gap: 10px; margin: 6px 0 10px 0; }
    .stat-card {
        flex: 1; min-width: 0;
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f4fd 100%);
        border-radius: 10px; padding: 10px 12px; text-align: center;
        border: 1px solid #d0e0f0;
        display: flex; flex-direction: column; justify-content: center;
    }
    .stat-card p { color: #6b7280; margin: 0; font-size: 12px; }
    .stat-card h3 { color: #1a1a2e; margin: 4px 0 0 0; font-size: 16px;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

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
    _dir = os.path.dirname(filepath)
    _raw_pq = os.path.join(_dir, '.cache_raw.parquet')
    _clean_pq = os.path.join(_dir, '.cache_cleaned.parquet')

    # Parquet 缓存: 首次从 Excel 读取后存为 Parquet，后续直接读 Parquet (快 10-20x)
    if os.path.exists(_raw_pq) and os.path.exists(_clean_pq):
        raw_df = pd.read_parquet(_raw_pq)
        cleaned_df = pd.read_parquet(_clean_pq)
    else:
        raw_df = load_raw_data(filepath)
        cleaned_df = clean_data(raw_df)
        try:
            for col in raw_df.select_dtypes(include=['object']).columns:
                raw_df[col] = raw_df[col].astype(str)
            for col in cleaned_df.select_dtypes(include=['object']).columns:
                cleaned_df[col] = cleaned_df[col].astype(str)
            raw_df.to_parquet(_raw_pq, index=False, engine='pyarrow')
            cleaned_df.to_parquet(_clean_pq, index=False, engine='pyarrow')
        except Exception:
            pass  # 如果 pyarrow 不可用则跳过缓存

    quality_report = get_data_quality_report(raw_df)
    cleaned_quality = get_data_quality_report(cleaned_df)
    cleaning_summary = get_cleaning_summary(raw_df, cleaned_df)
    rfm_df = calculate_rfm(cleaned_df)
    rfm_scored = add_rfm_scores(rfm_df)
    rfm_stats = get_rfm_stats(rfm_df)
    return raw_df, cleaned_df, quality_report, cleaned_quality, cleaning_summary, rfm_df, rfm_scored, rfm_stats

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_retail_II(1).xlsx")
raw_df, cleaned_df, quality_report, cleaned_quality, cleaning_summary, rfm_df, rfm_scored, rfm_stats = load_and_clean(DATA_PATH)

# ---- 页面 2 缓存函数 ----
@st.cache_data(show_spinner=False)
def filter_explore_data(_df, start_ts, end_ts):
    """按时间范围过滤数据并缓存"""
    df = _df[(_df['InvoiceDate'] >= start_ts) & (_df['InvoiceDate'] <= end_ts)].copy()
    return df

@st.cache_data(show_spinner=False)
def compute_distribution_stats(_df):
    """计算 Quantity / Price / Revenue 的分段统计"""
    results = {}
    configs = {
        'Quantity': {'bins': [0, 2, 5, 12, 24, 50, float('inf')],
                     'labels': ['1-2件', '3-5件', '6-12件', '13-24件', '25-50件', '50+件'],
                     'unit': '', 'fmt': '{:.0f}', 'numfmt': '{:.0f}'},
        'Price':    {'bins': [0, 1, 2, 5, 10, 20, float('inf')],
                     'labels': ['$0-1', '$1-2', '$2-5', '$5-10', '$10-20', '$20+'],
                     'unit': '美元', 'fmt': '{:.2f} 美元', 'numfmt': '{:.2f}'},
        'Revenue':  {'bins': [0, 5, 10, 20, 50, 100, float('inf')],
                     'labels': ['$0-5', '$5-10', '$10-20', '$20-50', '$50-100', '$100+'],
                     'unit': '美元', 'fmt': '{:.2f} 美元', 'numfmt': '{:.2f}'},
    }
    for col, cfg in configs.items():
        s = _df[col]
        counts = pd.cut(s, bins=cfg['bins'], labels=cfg['labels'], right=True).value_counts().sort_index()
        total = len(s)
        pcts = (counts / total * 100).round(1)
        quantiles = s.quantile([0.25, 0.5, 0.75, 0.90, 0.95])
        results[col] = {
            'labels': cfg['labels'],
            'counts': counts.tolist(),
            'pcts': pcts.tolist(),
            'p25': float(quantiles[0.25]), 'p50': float(quantiles[0.5]),
            'p75': float(quantiles[0.75]), 'p90': float(quantiles[0.90]),
            'p95': float(quantiles[0.95]),
            'mean': float(s.mean()), 'max': float(s.max()),
            'fmt': cfg['fmt'], 'unit': cfg['unit'], 'numfmt': cfg['numfmt'],
        }
    return results

@st.cache_data(show_spinner=False)
def compute_top_products(_df, top_n):
    """计算热销商品排名 (缓存)"""
    tp = _df.groupby(['StockCode', 'Description']).agg(
        收入=('Revenue', 'sum'),
        销量=('Quantity', 'sum'),
        订单数=('Invoice', 'nunique')
    ).sort_values('收入', ascending=False).head(top_n).reset_index()
    tp['商品名称'] = tp.apply(lambda r: f"{r['Description']}  ({r['StockCode']})", axis=1)
    return tp

@st.cache_data(show_spinner=False)
def compute_rfm_segments(_rfm_df):
    """计算 R/F/M 的分段统计 (缓存)"""
    configs = {
        'Recency': {
            'bins': [0, 7, 14, 30, 60, 120, 200, float('inf')],
            'labels': ['1-7天', '8-14天', '15-30天', '31-60天', '61-120天', '121-200天', '200+天'],
            'xaxis_title': '距上次购买 (天)', 'title': 'R - 最近购买间隔分布',
            'color': COLORS['primary'], 'fmt': '{:.0f}天', 'numfmt': '{:.0f}', 'unit': '天',
        },
        'Frequency': {
            'bins': [0, 1, 2, 5, 10, 20, float('inf')],
            'labels': ['1次', '2次', '3-5次', '6-10次', '11-20次', '20+次'],
            'xaxis_title': '订单数', 'title': 'F - 购买频率分布',
            'color': COLORS['danger'], 'fmt': '{:.0f}次', 'numfmt': '{:.0f}', 'unit': '次',
        },
        'Monetary': {
            'bins': [0, 200, 500, 1000, 2000, 5000, float('inf')],
            'labels': ['$0-200', '$200-500', '$500-1K', '$1K-2K', '$2K-5K', '$5K+'],
            'xaxis_title': '总消费 ($)', 'title': 'M - 消费金额分布',
            'color': COLORS['success'], 'fmt': '{:.0f} 美元', 'numfmt': '{:.0f}', 'unit': '美元',
        },
    }
    results = {}
    for col, cfg in configs.items():
        s = _rfm_df[col]
        counts = pd.cut(s, bins=cfg['bins'], labels=cfg['labels'], right=True).value_counts().sort_index()
        total = len(s)
        pcts = (counts / total * 100).round(1)
        quantiles = s.quantile([0.25, 0.5, 0.75, 0.90])
        results[col] = {
            'labels': cfg['labels'], 'counts': counts.tolist(), 'pcts': pcts.tolist(),
            'p25': float(quantiles[0.25]), 'p50': float(quantiles[0.5]),
            'p75': float(quantiles[0.75]), 'p90': float(quantiles[0.90]),
            'mean': float(s.mean()), 'max': float(s.max()),
            'xaxis_title': cfg['xaxis_title'], 'title': cfg['title'],
            'color': cfg['color'], 'fmt': cfg['fmt'],
            'numfmt': cfg['numfmt'], 'unit': cfg['unit'],
        }
    return results

def _fmt_val(info, v):
    """格式化单个值: 数字+单位"""
    num = info['numfmt'].format(v)
    unit = info.get('unit', '')
    if not unit:
        return num
    if unit in ('天', '次'):
        return num + unit
    return num + ' ' + unit

def _fmt_range(info, a, b):
    """格式化范围: 数字~数字+单位(单位只出现一次)"""
    num = f"{info['numfmt'].format(a)} ~ {info['numfmt'].format(b)}"
    unit = info.get('unit', '')
    if not unit:
        return num
    if unit in ('天', '次'):
        return num + unit
    return num + ' ' + unit

def stat_cards_row(items):
    """渲染一行紧凑等高统计卡片, items = [(label, value), ...]"""
    html = '<div class="stat-row">'
    for label, val in items:
        html += f'<div class="stat-card"><p>{label}</p><h3>{val}</h3></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_prepare_and_elbow(_rfm_df, method, winsorize_pct, k_start, k_end):
    """特征工程 + 最优 K 搜索 (缓存, 避免每次交互重跑)"""
    scaled, feature_names, scaler, transformed_df = prepare_features(
        _rfm_df, method=method, winsorize_pct=winsorize_pct)
    elbow = find_optimal_k(scaled, k_range=range(k_start, k_end + 1))
    return scaled, feature_names, scaler, transformed_df, elbow


@st.cache_data(show_spinner=False)
def cached_run_kmeans(_rfm_df, n_clusters, method, winsorize_pct):
    """K-Means 聚类 (缓存, 避免每次交互重跑)"""
    return run_kmeans(_rfm_df, n_clusters=n_clusters, method=method,
                      winsorize_pct=winsorize_pct)

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
st.sidebar.markdown(f"**📂 数据集:** Online Retail II")
st.sidebar.markdown(f"**📅 {cleaning_summary['date_range_start']} ~ {cleaning_summary['date_range_end']}**")
st.sidebar.markdown(f"**📦 清洗后:** {cleaning_summary['cleaned_rows']:,} 条 · {cleaning_summary['unique_customers']:,} 客户")

# ============================================================
# 页面 1: 数据概览
# ============================================================
if page == "📈 数据概览":
    st.title("📈 数据概览")
    st.markdown("UCI Online Retail II — 英国在线零售商店交易数据")

    # ---- 数据基础探索 ----
    st.subheader("📋 数据理解与基础探索")
    st.markdown("""
    **数据集来源**: UCI 机器学习仓库 — Online Retail II (ID: 502)，包含一家英国在线零售商 2009.12 至 2010.12 的全部交易记录。
    **8 个字段**: Invoice (发票号), StockCode (商品编码), Description (商品描述), Quantity (数量), InvoiceDate (日期), Price (单价), Customer ID (客户ID), Country (国家)。
    """)

    # 数据规模快速概览
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.metric("原始记录数", f"{cleaning_summary['raw_rows']:,}")
    with col2:
        st.markdown(
            f'<div class="date-card"><p>时间跨度</p><h3>{cleaning_summary["date_range_start"]} ~ {cleaning_summary["date_range_end"]}</h3></div>',
            unsafe_allow_html=True)
    with col3:
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

    # 清洗前后对比图
    st.markdown("#### 📊 清洗前后质量问题对比")
    _issue_labels = ['取消订单', '缺失客户ID', '负数量', '零/负价格', '精确重复', '非商品编码']
    _before = [quality_report['cancelled_orders'], quality_report['missing_customer_id'],
               quality_report['negative_quantity'], quality_report['zero_neg_price'],
               quality_report['exact_duplicates'], quality_report['special_stockcodes']]
    _after = [cleaned_quality['cancelled_orders'], cleaned_quality['missing_customer_id'],
              cleaned_quality['negative_quantity'], cleaned_quality['zero_neg_price'],
              cleaned_quality['exact_duplicates'], cleaned_quality['special_stockcodes']]
    _compare_df = pd.DataFrame({
        '问题类型': _issue_labels * 2,
        '数量': _before + _after,
        '阶段': ['清洗前'] * 6 + ['清洗后'] * 6,
    })
    fig_compare = px.bar(_compare_df, x='问题类型', y='数量', color='阶段',
                         barmode='group',
                         color_discrete_map={'清洗前': COLORS['danger'], '清洗后': COLORS['success']},
                         labels={'数量': '问题数量 (条)', '问题类型': '', '阶段': ''})
    fig_compare.update_traces(textposition='outside', textfont=dict(size=11))
    fig_compare.update_layout(
        **CHART_LAYOUT, height=380,
        title="各类型质量问题: 清洗前 vs 清洗后",
        xaxis=dict(tickfont=dict(size=12)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

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

    # ---- 本页面时间筛选器 ----
    with st.expander("📅 时间范围筛选 (仅影响本页图表)", expanded=False):
        _date_min = pd.Timestamp(cleaned_df['InvoiceDate'].min())
        _date_max = pd.Timestamp(cleaned_df['InvoiceDate'].max())
        _months = pd.period_range(_date_min, _date_max, freq='M')
        _fmt = lambda o: str(pd.Period(ordinal=o, freq='M'))
        _ms, _me = _date_min.to_period('M').ordinal, _date_max.to_period('M').ordinal
        c1, c2 = st.columns(2)
        with c1:
            _sel_s = st.selectbox("起始月", list(range(_ms, _me + 1)),
                                   index=0, format_func=_fmt, key="exp_s")
        with c2:
            _sel_e = st.selectbox("结束月", list(range(_ms, _me + 1)),
                                   index=len(_months) - 1, format_func=_fmt, key="exp_e")
        if _sel_s > _sel_e:
            _sel_s, _sel_e = _sel_e, _sel_s
        _fs = pd.Period(ordinal=_sel_s, freq='M').start_time
        _fe = pd.Period(ordinal=_sel_e, freq='M').end_time

    explore_df = filter_explore_data(cleaned_df, _fs, _fe)
    if len(explore_df) == 0:
        st.warning("⚠️ 所选时间范围内无数据。")
        st.stop()
    st.caption(f"当前范围: **{len(explore_df):,}** 条交易 · "
               f"**{explore_df['InvoiceDate'].min().date()}** ~ **{explore_df['InvoiceDate'].max().date()}**")

    # ---- 字段值分布 (方案C: 分段统计图) ----
    st.subheader("📊 字段值分布")
    st.caption("按业务含义分段统计，直观展示各区间订单数量与占比。")

    dist_stats = compute_distribution_stats(explore_df)

    _seg_configs = {
        'Quantity': {'title': '订单数量分布', 'xaxis_title': '购买数量',
                     'color': COLORS['primary'], 'icon': '📦'},
        'Price':    {'title': '单价分布', 'xaxis_title': '单价 (美元 $)',
                     'color': COLORS['secondary'], 'icon': '💲'},
        'Revenue':  {'title': '单笔交易收入分布', 'xaxis_title': '收入 (美元 $)',
                     'color': COLORS['success'], 'icon': '💰'},
    }

    tab1, tab2, tab3 = st.tabs(["📦 数量 (Quantity)", "💲 单价 (Price)", "💰 收入 (Revenue)"])

    for tab, col in zip([tab1, tab2, tab3], ['Quantity', 'Price', 'Revenue']):
        with tab:
            info = dist_stats[col]
            cfg = _seg_configs[col]

            # 统计卡片 (紧凑等高)
            stat_cards_row([
                ('中位数', _fmt_val(info, info['p50'])),
                ('均值', _fmt_val(info, info['mean'])),
                ('中间50%范围', _fmt_range(info, info['p25'], info['p75'])),
                ('前10%阈值', _fmt_val(info, info['p90'])),
            ])

            # 分段柱状图
            fig_seg = px.bar(
                x=info['labels'], y=info['counts'],
                color=info['counts'],
                color_continuous_scale=[cfg['color'] + '66', cfg['color']],
                labels={'x': cfg['xaxis_title'], 'y': '订单数'},
            )
            # 在柱子上方标注百分比
            fig_seg.update_traces(
                text=[f"{p}%" for p in info['pcts']],
                textposition='outside',
                textfont=dict(size=13, color='#333'),
                hovertemplate=f'{cfg["xaxis_title"]}'+'=%{x}<br>订单数=%{y:,.0f}<br>占比=%{text}<extra></extra>'
            )
            fig_seg.update_layout(
                **CHART_LAYOUT, height=400, title=cfg['title'],
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(tickfont=dict(size=13)),
                yaxis=dict(title='订单数'),
            )
            st.plotly_chart(fig_seg, use_container_width=True)

            st.markdown(
                f"下四分位: **{_fmt_val(info, info['p25'])}** · "
                f"中位数: **{_fmt_val(info, info['p50'])}** · "
                f"上四分位: **{_fmt_val(info, info['p75'])}** · "
                f"前5%值: **{_fmt_val(info, info['p95'])}** · "
                f"最大值: {_fmt_val(info, info['max'])}"
            )

    # ---- 热销商品 ----
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("🏆 热销商品排名 (按收入)")
    top_n = st.slider("显示数量", min_value=5, max_value=30, value=20, step=5, key="top_n")
    top_products = compute_top_products(explore_df, top_n)
    fig_top = px.bar(top_products, x='收入', y='商品名称', orientation='h',
                     color='收入', color_continuous_scale='Viridis',
                     hover_data=['StockCode', '销量', '订单数'])
    fig_top.update_layout(**CHART_LAYOUT, height=max(350, top_n * 32), showlegend=False,
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

    # R/F/M 核心指标 (紧凑表格, R/F/M 行标签清晰)
    rfm_seg = compute_rfm_segments(rfm_df)
    _rows = []
    for _dim, _lab in [('Recency', '**R** · 最近购买间隔'),
                       ('Frequency', '**F** · 购买频率'),
                       ('Monetary', '**M** · 消费金额')]:
        _i = rfm_seg[_dim]
        _rows.append(f"| {_lab} | {_fmt_val(_i, _i['p50'])} | {_fmt_val(_i, _i['mean'])} | "
                     f"{_fmt_val(_i, _i['p75'])} | {_fmt_val(_i, _i['p90'])} |")
    st.markdown(
        "| 维度 | 中位数 | 均值 | 上四分位 | 前10%阈值 |\n"
        "|:---|:---:|:---:|:---:|:---:|\n" + "\n".join(_rows)
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # R/F/M 分布 (分段统计图)
    col1, col2, col3 = st.columns(3)
    for mc, dim in zip([col1, col2, col3], ['Recency', 'Frequency', 'Monetary']):
        with mc:
            info = rfm_seg[dim]
            fig_seg = px.bar(
                x=info['labels'], y=info['counts'],
                color=info['counts'],
                color_continuous_scale=[info['color'] + '66', info['color']],
                labels={'x': info['xaxis_title'], 'y': '客户数'},
            )
            fig_seg.update_traces(
                text=[f"{p}%" for p in info['pcts']],
                textposition='outside',
                textfont=dict(size=12, color='#333'),
                hovertemplate=f'{info["xaxis_title"]}'+'=%{x}<br>客户数=%{y:,.0f}<br>占比=%{text}<extra></extra>'
            )
            fig_seg.update_layout(
                **CHART_LAYOUT, height=380, title=info['title'],
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(tickfont=dict(size=11)),
                yaxis=dict(title='客户数'),
            )
            fig_seg.update_layout(margin=dict(l=50, r=10, t=50, b=60))
            st.plotly_chart(fig_seg, use_container_width=True)

    st.caption("💡 **解读**: R (最近购买间隔) 中位数 52 天，大多数客户在 2 个月内有购买行为。F (购买频率) 以 1-2 次为主，M (消费金额) 以 200-500 美元为主——大多数客户为低频低消费群体，少量高频高消费客户拉高了均值。这种偏态分布是后续需要做对数变换的原因。")

    # RFM 相关性矩阵
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    col_corr1, col_corr2 = st.columns([1, 2])
    with col_corr1:
        st.subheader("🔗 RFM 相关性矩阵")
        st.markdown("""
        F (频率) 和 M (金额) 呈中等正相关 (r≈0.65)，说明买得多的客户也倾向于花得多。
        这一相关性是后续聚类中将 F/M 合并为 composite 特征的理论依据。
        """)
    with col_corr2:
        corr = rfm_df[['Recency', 'Frequency', 'Monetary']].corr()
        corr_display = corr.copy()
        corr_display.index = ['R-最近购买', 'F-购买频率', 'M-消费金额']
        corr_display.columns = ['R-最近购买', 'F-购买频率', 'M-消费金额']
        fig_corr = px.imshow(corr_display.values,
                             x=corr_display.columns.tolist(),
                             y=corr_display.index.tolist(),
                             color_continuous_scale='RdBu_r',
                             zmin=-1, zmax=1,
                             text_auto='.2f',
                             aspect='auto',
                             labels={'color': '相关系数'})
        fig_corr.update_layout(**CHART_LAYOUT, height=340)
        st.plotly_chart(fig_corr, use_container_width=True)

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

    with st.expander("📋 查看分类依据 (评分规则)"):
        st.markdown("""
**评分方法**: 将 R / F / M 各按分位数分为 1~5 档 (5 为最优)：

| 得分 | R (最近购买) | F (购买频率) | M (消费金额) |
|:---:|:---:|:---:|:---:|
| **5** | 最近 20% 的客户 | 最频繁的 20% | 消费最高的 20% |
| **4** | 次近 20% | 次频繁 20% | 次高 20% |
| **3** | 中间 20% | 中间 20% | 中间 20% |
| **2** | 次远 20% | 次低频 20% | 次低 20% |
| **1** | 最远 20% | 最低频 20% | 最低 20% |

**分群规则**:

| 客户分群 | R 分 | F 分 | M 分 | 含义 |
|:---|:---:|:---:|:---:|:---|
| 冠军客户 | ≥4 | ≥4 | ≥4 | 近期活跃、高频高消费的核心客户 |
| 忠诚客户 | ≥3 | ≥3 | — | 持续购买的稳定客户 |
| 新客户 | ≥4 | ≤2 | — | 刚来不久、尚未形成购买习惯 |
| 流失风险 | ≤2 | ≥3 | ≥3 | 曾经活跃但近期不再购买 |
| 流失客户 | ≤2 | ≤2 | ≤2 | 长期不活跃、低消费 |
| 潜力客户 | ≥3 | 1~2 | — | 有一定活跃度、可挖掘价值 |
| 沉睡客户 | ≤2 | — | — | 不活跃但未完全流失 |
| 待开发 | 其他 | — | — | 尚未明确归类的客户 |
        """)

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

    # 特征准备 + 最优 K 搜索 (缓存)
    scaled, feature_names, scaler, transformed_df, elbow_result = cached_prepare_and_elbow(
        rfm_df, cluster_method, winsorize_pct, k_range_start, k_range_end)

    # 最优 K 分析
    st.subheader("📐 第一步: 确定最优聚类数 K")
    col1, col2 = st.columns(2)

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
    result = cached_run_kmeans(rfm_df, selected_k, cluster_method, winsorize_pct)
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

    # 3D 分群散点图 (按群单独查看 + 大色块图例 + 簇中心标注)
    st.subheader("🌐 客户 3D 分群可视化")
    plot_df = clustered_df.copy()

    _cids = sorted(labels.keys())
    _all_opts = [f"C{c}: {labels[c]['name']}" for c in _cids]
    _def_opt = next((o for o in _all_opts if '重要价值' in o), _all_opts[0])

    # 各轴截断到 99 分位, 防止极端离群值把主云团压成一小团
    _cap = {c: float(plot_df[c].quantile(0.99)) for c in ['Recency', 'Frequency', 'Monetary']}
    for _c in _cap:
        plot_df[_c] = plot_df[_c].clip(upper=_cap[_c])

    # 每个客户群一条 trace (点 + 该群中心菱形/群名), 底部图例点击即可显示/隐藏;
    # 默认仅显示一个群, 其余 legendonly (点了才显示); 图例切换为浏览器端操作, 不重跑后端
    fig_3d = go.Figure()
    for _i, _cid in enumerate(_cids):
        _opt = _all_opts[_i]
        _sub = plot_df[plot_df['Cluster'] == _cid]
        _n = len(_sub)
        _cx = float(_sub['Recency'].mean())
        _cy = float(_sub['Frequency'].mean())
        _cz = float(_sub['Monetary'].mean())
        fig_3d.add_trace(go.Scatter3d(
            x=list(_sub['Recency']) + [_cx],
            y=list(_sub['Frequency']) + [_cy],
            z=list(_sub['Monetary']) + [_cz],
            mode='markers+text',
            marker=dict(
                size=[5] * _n + [14],
                symbol=['circle'] * _n + ['diamond'],
                color=COLORS['palette'][_i],
                opacity=0.75,
                line=dict(width=0.5, color='white'),
            ),
            text=[''] * _n + [labels[_cid]['name']],
            textposition='top center',
            textfont=dict(size=16, color='#111827'),
            customdata=[[str(v)] for v in _sub['Customer ID']] + [[labels[_cid]['name'] + ' (群中心)']],
            hovertemplate='客户 %{customdata[0]}<br>R=%{x:.0f} 天 / F=%{y:.0f} 次 / M=%{z:,.0f} 美元<extra>' + _opt + '</extra>',
            name=_opt,
            showlegend=True,
            visible=True if _opt == _def_opt else 'legendonly',
        ))

    fig_3d.update_layout(**CHART_LAYOUT, height=680)
    fig_3d.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        scene=dict(
            xaxis=dict(backgroundcolor='#fafbfc', gridcolor='#e5e7eb',
                       title='R-最近购买 (天)', range=[0, _cap['Recency']]),
            yaxis=dict(backgroundcolor='#fafbfc', gridcolor='#e5e7eb',
                       title='F-购买频率', range=[0, _cap['Frequency']]),
            zaxis=dict(backgroundcolor='#fafbfc', gridcolor='#e5e7eb',
                       title='M-消费金额 (美元)', range=[0, _cap['Monetary']]),
            bgcolor='white',
            aspectmode='manual',
            aspectratio=dict(x=1.35, y=1.35, z=1.0),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
        legend=dict(orientation='h', yanchor='top', y=-0.02, xanchor='center', x=0.5,
                    font=dict(size=15), itemsizing='constant'),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # 2D 特征空间散点图 (聚类实际发生的空间)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("📍 特征空间 2D 可视化")
    st.caption("下图展示聚类**实际使用的特征空间** (非原始 R/F/M)。支持框选 (lasso/box) 查看选中客户的详细信息。")

    plot_2d = transformed_df.reset_index(drop=True).copy()
    # transformed_df 仅含工程特征列, 需按行序从 clustered_df 补回 Cluster / Customer ID / 原始 R F M
    _src = clustered_df[['Customer ID', 'Cluster', 'Recency', 'Frequency', 'Monetary']].reset_index(drop=True)
    for _col in _src.columns:
        if _col not in plot_2d.columns:
            plot_2d[_col] = _src[_col]
    plot_2d['聚类标签'] = plot_2d['Cluster'].map(lambda c: f"C{c}: {labels[c]['name']}")

    if len(feature_names) == 2:
        _fx, _fy = feature_names[0], feature_names[1]
        _xl = 'R-时效性 (排名)' if _fx == 'R_rank' else _fx
        _yl = 'RFM-参与度 (综合排名)' if _fy == 'RFM_composite' else _fy
    else:
        _fx, _fy = feature_names[0], feature_names[1]
        _xl, _yl = _fx, _fy

    fig_2d = px.scatter(
        plot_2d, x=_fx, y=_fy,
        color='聚类标签',
        hover_data=['Customer ID', 'Recency', 'Frequency', 'Monetary'],
        opacity=0.7,
        color_discrete_sequence=COLORS['palette'],
        labels={_fx: _xl, _fy: _yl},
    )
    fig_2d.update_layout(**CHART_LAYOUT, height=500,
                         title="聚类特征空间散点图 (支持框选交互)")
    selection = st.plotly_chart(fig_2d, use_container_width=True, selection_mode="points",
                                on_select="rerun", key="scatter_2d")

    # 展示框选客户详情
    sel_events = selection.get('selection', {}).get('points', [])
    if sel_events:
        sel_ids = [p.get('customdata', [None])[0] for p in sel_events if p.get('customdata')]
        sel_customers = rfm_df[rfm_df['Customer ID'].isin(sel_ids)].sort_values('Monetary', ascending=False)
        if len(sel_customers) > 0:
            with st.expander(f"📋 已选中 {len(sel_customers)} 位客户 — 点击展开详情", expanded=True):
                display_sel = sel_customers[['Customer ID', 'Recency', 'Frequency', 'Monetary']].copy()
                display_sel.columns = ['客户 ID', 'R (天)', 'F (次)', 'M ($)']
                st.dataframe(display_sel, use_container_width=True, hide_index=True, height=300)

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
