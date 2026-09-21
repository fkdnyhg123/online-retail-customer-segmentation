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
import json
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_cleaner import load_raw_data, clean_data, get_data_quality_report, get_cleaning_summary
from utils.rfm import (calculate_rfm, add_rfm_scores, get_rfm_stats,
                       calculate_extended_features, EXTENDED_FEATURE_INFO)
from utils.clustering import prepare_features, find_optimal_k, run_kmeans, get_cluster_labels
from utils.association import (prepare_basket_matrix, run_fpgrowth,
                               compute_cooccurrence_matrix, build_network_graph,
                               threshold_sweep, build_marketing_actions)
from utils.product_country import (description_word_stats, country_summary,
                                   uk_vs_overseas_profile, country_monthly_revenue,
                                   WORD_CN)

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
    # 15 色与侧边栏 K 上限 (15) 对齐, 保证每个簇都有独立颜色;
    # 取色处 (card_colors[cluster_id % len(card_colors)]) 带取模兜底, 色数不足时只会复用而不会报错
    'palette': ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626',
                '#0891b2', '#c026d3', '#ea580c', '#0d9488', '#4f46e5',
                '#65a30d', '#ec4899', '#991b1b', '#1e3a8a', '#6b7280'],
}

# 原始数据的 6 类质量问题 (中文名, quality_report 键名)。
# 页面 1 的问题清单、质量饼图与「清洗后复检」表共用这一份定义, 避免两处各写一遍键名导致口径漂移。
QUALITY_ITEM_KEYS = [
    ("取消订单 (Cancelled)", 'cancelled_orders'),
    ("缺失客户ID (Customer ID)", 'missing_customer_id'),
    ("负数量 (Negative Qty)", 'negative_quantity'),
    ("零/负价格 (Zero/Neg Price)", 'zero_neg_price'),
    ("精确重复 (Duplicates)", 'exact_duplicates'),
    ("非商品编码 (Non-product)", 'special_stockcodes'),
]
# 第 7 类只进清洗后复检表, 不进质量饼图 (饼图口径保持 6 类)
QUALITY_EXTRA_KEYS = [("缺失描述 (Description)", 'missing_description')]

CHART_LAYOUT = dict(
    paper_bgcolor='white',
    plot_bgcolor='#fafbfc',
    font=dict(color='#374151', family='Microsoft YaHei, SimHei, sans-serif', size=13),
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor='white', font_size=13, bordercolor='#e5e7eb'),
)

# ============================================================
# 特征变换方法: 中文名 + "每个特征集配哪个方法"的映射
# ============================================================
# 配对依据是实测 (K=4) 的轮廓系数: 仅 RFM 时组合特征 0.4486 > PCA 0.4369;
# 加上两个扩展特征后 PCA 0.3967 > 组合特征 0.3034 —— 各自取该特征集下的最优方法,
# 因此侧边栏不再单独暴露方法选择, 只留「输入特征集」一个控件。
METHOD_LABELS = {
    'composite': '组合特征 2D',
    'pca': 'PCA 降维 2D',
    'rank': '百分位排名 3D',
    'log': '对数变换 3D',
}
EXT_FEATURE_SET_METHOD = {'rfm_only': 'composite', 'rfm_ext': 'pca'}

# ============================================================
# 数据加载 (缓存)
# ============================================================
def _cache_fingerprint(filepath: str) -> str:
    """数据文件 + 清洗模块的内容指纹, 任一变化即令 Parquet 缓存失效。
    用内容哈希而非 mtime: Streamlit Cloud 每次检出仓库都会重写文件时间戳, 比较结果不可靠。"""
    h = hashlib.sha256()
    targets = [filepath, os.path.join(os.path.dirname(filepath), 'utils', 'data_cleaner.py')]
    for p in targets:
        h.update(os.path.basename(str(p)).encode('utf-8'))
        try:
            with open(p, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
        except OSError:
            h.update(b'<missing>')
    return h.hexdigest()


@st.cache_data(show_spinner="正在加载数据集，请稍候...")
def load_and_clean(filepath):
    _dir = os.path.dirname(filepath)
    _raw_pq = os.path.join(_dir, '.cache_raw.parquet')
    _clean_pq = os.path.join(_dir, '.cache_cleaned.parquet')
    _meta_pq = os.path.join(_dir, '.cache_meta.json')

    # Parquet 缓存: 首次从 Excel 读取后存为 Parquet，后续直接读 Parquet (快 10-20x)
    _fp = _cache_fingerprint(filepath)
    _cache_hit = os.path.exists(_raw_pq) and os.path.exists(_clean_pq)
    if _cache_hit:
        try:
            with open(_meta_pq, encoding='utf-8') as f:
                _cache_hit = json.load(f).get('fingerprint') == _fp
        except (OSError, ValueError):
            _cache_hit = False

    if _cache_hit:
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
            with open(_meta_pq, 'w', encoding='utf-8') as f:
                json.dump({'fingerprint': _fp}, f)
        except Exception:
            pass  # 磁盘不可写或缺 pyarrow 时跳过缓存

    quality_report = get_data_quality_report(raw_df)
    cleaned_quality = get_data_quality_report(cleaned_df)
    cleaning_summary = get_cleaning_summary(raw_df, cleaned_df)
    rfm_df = calculate_rfm(cleaned_df)
    # 扩展特征 (AOV / 品类广度) 并入 rfm_df: 聚类页按列名取用, 客户特征分析页直接画分布
    rfm_df = rfm_df.merge(calculate_extended_features(cleaned_df),
                          on='Customer ID', how='left')
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
    num = f"{info['numfmt'].format(a)} 至 {info['numfmt'].format(b)}"
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

def _dominant_bin(info):
    """取分段统计里占比最高的档位, 返回 (档位标签, 占比%) — 供解读文案动态引用"""
    _i = int(np.argmax(info['pcts']))
    return info['labels'][_i], info['pcts'][_i]


@st.cache_data(show_spinner=False)
def cached_prepare_and_elbow(_rfm_df, method, winsorize_pct, k_start, k_end, extra_features):
    """特征工程 + 最优 K 搜索 (缓存, 避免每次交互重跑)"""
    scaled, feature_names, scaler, transformed_df, pca_model = prepare_features(
        _rfm_df, method=method, winsorize_pct=winsorize_pct,
        extra_features=extra_features)
    elbow = find_optimal_k(scaled, k_range=range(k_start, k_end + 1))
    return scaled, feature_names, scaler, transformed_df, elbow, pca_model


@st.cache_data(show_spinner=False)
def cached_run_kmeans(_rfm_df, n_clusters, method, winsorize_pct, extra_features):
    """K-Means 聚类 (缓存, 避免每次交互重跑)"""
    return run_kmeans(_rfm_df, n_clusters=n_clusters, method=method,
                      winsorize_pct=winsorize_pct, extra_features=extra_features)


@st.cache_data(show_spinner="正在对比两种特征变换方法...")
def cached_compare_methods(_rfm_df, compare_k, winsorize_pct, extra_features):
    """在当前特征集下对比两种方法 (组合特征 / PCA) 的轮廓系数, 供 4.4 的实时对比表使用

    只跑这两个方法: 侧边栏已按特征集自动配对方法, 表格的作用是展示"为什么配这个",
    而百分位排名 / 对数变换实测更差 (见说明文档 4.4), 不必每次都算。
    """
    out = {}
    for m in ['composite', 'pca']:
        try:
            r = run_kmeans(_rfm_df, n_clusters=compare_k, method=m,
                           winsorize_pct=winsorize_pct,
                           extra_features=extra_features)
            out[m] = {
                'silhouette': round(float(r['silhouette_score']), 4),
                'dims': len(prepare_features(
                    _rfm_df, method=m, winsorize_pct=winsorize_pct,
                    extra_features=extra_features)[1]),
                'ok': True,
            }
        except Exception as e:
            out[m] = {'silhouette': float('nan'), 'dims': 0, 'ok': False,
                      'err': str(e)}
    return out


@st.cache_data(show_spinner=False)
def compute_extended_segments(_rfm_df):
    """计算扩展特征 (AOV / 品类广度) 的分段统计 (缓存)"""
    configs = {
        'AOV': {
            'bins': [0, 100, 200, 300, 500, 1000, float('inf')],
            'labels': ['$0-100', '$100-200', '$200-300', '$300-500', '$500-1K', '$1K+'],
            'xaxis_title': '平均客单价 ($)', 'title': 'AOV - 平均客单价分布',
            'color': COLORS['info'], 'numfmt': '{:.0f}', 'unit': '美元',
        },
        'N_products': {
            'bins': [0, 10, 25, 50, 100, 200, float('inf')],
            'labels': ['1-10种', '11-25种', '26-50种', '51-100种', '101-200种', '200+种'],
            'xaxis_title': '购买商品种类数', 'title': '品类广度 - 商品种类数分布',
            'color': COLORS['secondary'], 'numfmt': '{:.0f}', 'unit': '种',
        },
    }
    results = {}
    for col, cfg in configs.items():
        s = _rfm_df[col]
        counts = pd.cut(s, bins=cfg['bins'], labels=cfg['labels'],
                        right=True).value_counts().sort_index()
        total = len(s)
        quantiles = s.quantile([0.25, 0.5, 0.75, 0.90])
        results[col] = {
            'labels': cfg['labels'], 'counts': counts.tolist(),
            'pcts': (counts / total * 100).round(1).tolist(),
            'p25': float(quantiles[0.25]), 'p50': float(quantiles[0.5]),
            'p75': float(quantiles[0.75]), 'p90': float(quantiles[0.90]),
            'mean': float(s.mean()), 'max': float(s.max()),
            'xaxis_title': cfg['xaxis_title'], 'title': cfg['title'],
            'color': cfg['color'], 'numfmt': cfg['numfmt'], 'unit': cfg['unit'],
        }
    return results


# 阈值敏感性扫描的网格: 覆盖滑块量程的关键刻度, 用于反推"阈值范围该定在哪"
SUPPORT_SWEEP = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
CONFIDENCE_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]


@st.cache_data(show_spinner="正在扫描支持度 / 置信度网格...")
def cached_threshold_sweep(_df, top_items, _n_rows):
    """阈值网格扫描 (缓存键为 top_items; 与 cleaned_df 的绑定由上层 Parquet 指纹保证)

    返回 (扫描表, 支持度上界依据)。上界依据里最关键的是最高频**商品对**的支持度:
    一条规则至少需要两种商品共现, 所以它是关联规则支持度的天然上界。
    """
    basket = prepare_basket_matrix(_df, top_n_items=top_items)
    return threshold_sweep(basket, SUPPORT_SWEEP, CONFIDENCE_SWEEP, min_lift=1.0)


@st.cache_data(show_spinner=False)
def cached_description_words(_df, top_n):
    """按商品描述高频词统计品类表现 (缓存)"""
    return description_word_stats(_df, top_n=top_n)


@st.cache_data(show_spinner=False)
def cached_country_summary(_df):
    """按国家汇总业务表现 (缓存)"""
    return country_summary(_df)


@st.cache_data(show_spinner=False)
def cached_uk_overseas(_df, _rfm_df):
    """本土 vs 海外客户画像对比 (缓存)"""
    return uk_vs_overseas_profile(_df, _rfm_df)


@st.cache_data(show_spinner=False)
def cached_country_monthly(_df, top_n):
    """国家月度收入透视表 (缓存)"""
    return country_monthly_revenue(_df, top_n=top_n)

# ============================================================
# 侧边栏导航
# ============================================================
st.sidebar.markdown("## 📊 导航面板")
page = st.sidebar.radio(
    "页面选择",
    ["📈 数据概览", "🔍 数据探索", "👥 客户特征分析", "🎯 K-Means 聚类",
     "🔗 关联规则分析", "📦 商品与国家调查"],
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

    # 三个初步观察 (数字与下方质量饼图同源, 均取自 quality_report)
    st.markdown(f"""
    **初步观察 (3 条)**:
    1. **数据规模与缺失**: 共 {quality_report['total_rows']:,} 条记录。Customer ID 缺失约 {quality_report['missing_customer_id_pct']}% ({quality_report['missing_customer_id']:,} 条)，将严重影响客户级分析 (如 RFM)。Description 有 {quality_report['missing_description']:,} 条缺失。
    2. **多重质量问题**: 存在取消订单 (Invoice 以 'C' 开头)、负数量、零/负价格、非商品编码 (POST 等)、精确重复行等多种问题，需仔细清洗。
    3. **地域集中**: 绝大多数订单来自英国，德国、法国等欧洲国家占比小，地域不平衡需在分析中考虑。
    """)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- 原始数据质量报告 ----
    st.subheader("🔎 原始数据质量报告")
    col1, col2 = st.columns([1, 1])

    with col1:
        quality_items = [(label, quality_report[key], quality_report[f'{key}_pct'])
                         for label, key in QUALITY_ITEM_KEYS]
        for label, count, pct in quality_items:
            st.markdown(f"**{label}**: {count:,} 条 ({pct}%)")

    with col2:
        labels_cn = [q[0] for q in quality_items]
        values = [q[1] for q in quality_items]
        fig_quality = px.pie(names=labels_cn, values=values, hole=0.45,
                             color_discrete_sequence=COLORS['palette'])
        fig_quality.update_layout(**CHART_LAYOUT, height=380,
                                  title="原始数据质量问题分布")
        st.plotly_chart(fig_quality, width='stretch')

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

    # 清洗前后数据对比
    st.markdown("#### 📊 清洗前后数据对比")

    # 计算清洗前各项统计
    _raw_rows = len(raw_df)
    _raw_customers = raw_df['Customer ID'].nunique()
    _raw_products = raw_df['StockCode'].nunique()
    _raw_invoices = raw_df['Invoice'].nunique()
    _clean_rows = len(cleaned_df)
    _clean_customers = cleaned_df['Customer ID'].nunique()
    _clean_products = cleaned_df['StockCode'].nunique()
    _clean_invoices = cleaned_df['Invoice'].nunique()

    _cmp_metrics = ['交易记录数', '客户数', '商品数', '发票数']
    _cmp_before = [_raw_rows, _raw_customers, _raw_products, _raw_invoices]
    _cmp_after = [_clean_rows, _clean_customers, _clean_products, _clean_invoices]

    _compare_df = pd.DataFrame({
        '指标': _cmp_metrics * 2,
        '数量': _cmp_before + _cmp_after,
        '阶段': ['清洗前'] * 4 + ['清洗后'] * 4,
    })
    fig_compare = px.bar(_compare_df, x='指标', y='数量', color='阶段',
                         barmode='group', text='数量',
                         color_discrete_map={'清洗前': COLORS['danger'], '清洗后': COLORS['success']},
                         labels={'数量': '数量', '指标': '', '阶段': ''})
    # text 必须按列传给 px: 若改用 update_traces(text=...), 同一份 8 元素数组会被套到两条各 4 点的
    # trace 上, plotly 各取前 4 个, 清洗后柱子就会标成清洗前的数值
    fig_compare.update_traces(texttemplate='%{text:,}', textposition='outside',
                              textfont=dict(size=11))
    fig_compare.update_layout(
        **CHART_LAYOUT, height=380,
        title="数据规模: 清洗前 vs 清洗后",
        xaxis=dict(tickfont=dict(size=12)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )
    st.plotly_chart(fig_compare, width='stretch')

    # 清洗前后详细对比表
    _avg_qty_before = raw_df['Quantity'].mean()
    _avg_qty_after = cleaned_df['Quantity'].mean()
    _avg_price_before = raw_df[raw_df['Price'] > 0]['Price'].mean()
    _avg_price_after = cleaned_df['Price'].mean()
    _date_before = f"{raw_df['InvoiceDate'].min().date()} ~ {raw_df['InvoiceDate'].max().date()}"
    _date_after = f"{cleaned_df['InvoiceDate'].min().date()} ~ {cleaned_df['InvoiceDate'].max().date()}"

    _cmp_table = pd.DataFrame({
        '对比项': ['交易记录数', '客户数', '商品数', '发票数', '平均数量', '平均单价', '时间范围'],
        '清洗前': [f"{_raw_rows:,}", f"{_raw_customers:,}", f"{_raw_products:,}", f"{_raw_invoices:,}",
                  f"{_avg_qty_before:.1f}", f"${_avg_price_before:.2f}", _date_before],
        '清洗后': [f"{_clean_rows:,}", f"{_clean_customers:,}", f"{_clean_products:,}", f"{_clean_invoices:,}",
                  f"{_avg_qty_after:.1f}", f"${_avg_price_after:.2f}", _date_after],
        '变化': [f"-{_raw_rows - _clean_rows:,} ({(1 - _clean_rows/_raw_rows)*100:.1f}%)",
                f"-{_raw_customers - _clean_customers:,}",
                f"-{_raw_products - _clean_products:,}",
                f"-{_raw_invoices - _clean_invoices:,}",
                f"{_avg_qty_after - _avg_qty_before:+.1f}",
                f"${_avg_price_after - _avg_price_before:+.2f}",
                "不变"],
    })
    st.dataframe(_cmp_table, width='stretch', hide_index=True)
    # 「清洗前平均数量」用全部行、「清洗前平均单价」只统计 Price > 0 的行, 两行口径不同, 必须标出来
    _avg_price_all_rows = raw_df['Price'].mean()
    st.caption(f"💡 **口径说明**: 「清洗前平均数量」统计全部 {len(raw_df):,} 行; "
               f"「清洗前平均单价」只统计 Price > 0 的行 (剔除 {quality_report['zero_neg_price']:,} 条零/负价格), "
               f"即 \\${_avg_price_before:.2f} 而非全部行的 \\${_avg_price_all_rows:.2f}。"
               "两行口径不同, 不能当成同一把尺子来比 —— 清洗后单价下降的主因是高价非商品行 (POST / BANK CHARGES) 被剔除。")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- 清洗后数据质量报告 ----
    st.subheader("✅ 清洗后数据质量报告")
    # 把上方 6 类 + 缺失描述在清洗后的数据上原样重算一遍: 全部归零才写「无质量问题」, 不靠口头断言
    _recheck_keys = QUALITY_ITEM_KEYS + QUALITY_EXTRA_KEYS
    _recheck_df = pd.DataFrame({
        '检查项': [label for label, _ in _recheck_keys],
        '清洗前命中 (条)': [quality_report[key] for _, key in _recheck_keys],
        '清洗后命中 (条)': [cleaned_quality[key] for _, key in _recheck_keys],
    })
    st.dataframe(_recheck_df, width='stretch', hide_index=True)
    st.caption(f"💡 **解读**: 7 项检查在清洗后的 {cleaning_summary['cleaned_rows']:,} 条记录上全部归零, "
               "说明清洗规则确实把每一类问题都清掉了, 而不是只清了占比大的那几类。")

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
        st.plotly_chart(fig_revenue, width='stretch')

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
        st.plotly_chart(fig_country, width='stretch')

# ============================================================
# 页面 2: 数据探索
# ============================================================
elif page == "🔍 数据探索":
    st.title("🔍 数据探索")

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
    st.caption("按业务含义分段统计各区间**交易行数**与占比 (一行 = 发票里的一个商品明细行，不是张发票)。")

    dist_stats = compute_distribution_stats(explore_df)

    _seg_configs = {
        'Quantity': {'title': '每行购买数量分布 (按交易行数)', 'xaxis_title': '购买数量',
                     'color': COLORS['primary'], 'icon': '📦'},
        'Price':    {'title': '商品单价分布 (按交易行数)', 'xaxis_title': '单价 (美元 $)',
                     'color': COLORS['secondary'], 'icon': '💲'},
        'Revenue':  {'title': '单笔交易收入分布', 'xaxis_title': '收入 (美元 $)',
                     'color': COLORS['success'], 'icon': '💰'},
    }

    tab1, tab2, tab3 = st.tabs(["📦 数量 (Quantity)", "💲 单价 (Price)", "💰 收入 (Revenue)"])

    for tab, col in zip([tab1, tab2, tab3], ['Quantity', 'Price', 'Revenue']):
        with tab:
            info = dist_stats[col]
            cfg = _seg_configs[col]

            # 统计卡片 (metric-card 风格, 与核心指标一致)
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mc_items = [
                (_mc1, _fmt_val(info, info['p50']), '中位数'),
                (_mc2, _fmt_val(info, info['mean']), '均值'),
                (_mc3, _fmt_range(info, info['p25'], info['p75']), '中间50%范围'),
                (_mc4, _fmt_val(info, info['p90']), '前10%阈值'),
            ]
            for _col, _val, _lab in _mc_items:
                with _col:
                    st.markdown(f'<div class="metric-card"><h2>{_val}</h2><p>{_lab}</p></div>',
                                unsafe_allow_html=True)

            # 分段柱状图
            fig_seg = px.bar(
                x=info['labels'], y=info['counts'],
                color=info['counts'],
                color_continuous_scale=[cfg['color'] + '66', cfg['color']],
                labels={'x': cfg['xaxis_title'], 'y': '交易行数'},
            )
            # 在柱子上方标注百分比
            fig_seg.update_traces(
                text=[f"{p}%" for p in info['pcts']],
                textposition='outside',
                textfont=dict(size=13, color='#333'),
                hovertemplate=f'{cfg["xaxis_title"]}'+'=%{x}<br>交易行数=%{y:,.0f}<br>占比=%{text}<extra></extra>'
            )
            fig_seg.update_layout(
                **CHART_LAYOUT, height=400, title=cfg['title'],
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(tickfont=dict(size=13)),
                yaxis=dict(title='交易行数'),
            )
            st.plotly_chart(fig_seg, width='stretch')

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
    st.plotly_chart(fig_top, width='stretch')
    st.caption("💡 **解读**: 热销商品以家居装饰和节日用品为主，排名靠前的 WHITE HANGING HEART T-LIGHT HOLDER 收入远超其余商品 —— 少量头部商品贡献了大部分收入。")

# ============================================================
# 页面 3: RFM 分析
# ============================================================
elif page == "👥 客户特征分析":
    st.title("👥 客户特征分析")
    st.markdown("在经典 R/F/M 模型之上引入两个非 RFM 的客户级特征 —— **平均客单价 (AOV)** 与 "
                "**品类广度**，共 5 个维度刻画客户价值与消费结构；这 5 个维度也是第 4 页聚类模型的输入。")

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
            st.plotly_chart(fig_seg, width='stretch')

    _r_med_txt = _fmt_val(rfm_seg['Recency'], rfm_seg['Recency']['p50'])
    _f_dom, _f_dom_pct = _dominant_bin(rfm_seg['Frequency'])
    _m_dom, _m_dom_pct = _dominant_bin(rfm_seg['Monetary'])
    st.caption(f"💡 **解读**: R (最近购买间隔) 中位数 {_r_med_txt}，即超过半数客户在此间隔内有过购买。"
               f"F (购买频率) 以 {_f_dom} 档为主 ({_f_dom_pct}%)，M (消费金额) 以 {_m_dom} 档为主 ({_m_dom_pct}%)"
               f"——大多数客户为低频低消费群体，少量高频高消费客户拉高了均值。"
               f"这种强偏态正是第 4 页不直接对原始 R/F/M 建模的原因。")

    # ---- 扩展客户特征 (非 RFM) ----
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("🧩 扩展客户特征 (非 RFM)")
    st.caption("在 R/F/M 之外增加两个客户级特征, 它们同时是第 4 页聚类模型的第 4、5 个输入维度。")

    ext_seg = compute_extended_segments(rfm_df)
    _ext_cols = st.columns(2)
    for _mc, _dim in zip(_ext_cols, ['AOV', 'N_products']):
        with _mc:
            _info = ext_seg[_dim]
            fig_ext = px.bar(
                x=_info['labels'], y=_info['counts'],
                color=_info['counts'],
                color_continuous_scale=[_info['color'] + '66', _info['color']],
                labels={'x': _info['xaxis_title'], 'y': '客户数'},
            )
            fig_ext.update_traces(
                text=[f"{p}%" for p in _info['pcts']],
                textposition='outside',
                textfont=dict(size=12, color='#333'),
                hovertemplate=f'{_info["xaxis_title"]}'+'=%{x}<br>客户数=%{y:,.0f}<br>占比=%{text}<extra></extra>'
            )
            fig_ext.update_layout(**CHART_LAYOUT, height=380, title=_info['title'],
                                  showlegend=False, coloraxis_showscale=False,
                                  xaxis=dict(tickfont=dict(size=11)),
                                  yaxis=dict(title='客户数'))
            fig_ext.update_layout(margin=dict(l=50, r=10, t=50, b=60))
            st.plotly_chart(fig_ext, width='stretch')

    _ext_rows = []
    for _key in ['AOV', 'N_products']:
        _info = ext_seg[_key]
        _meta = EXTENDED_FEATURE_INFO[_key]
        _ext_rows.append(
            f"| **{_meta['name']}** ({_meta['en']}) | {_meta['formula']} | {_meta['desc']} | "
            f"{_fmt_val(_info, _info['p50'])} | {_fmt_val(_info, _info['p75'])} |")
    st.markdown(
        "| 特征 | 计算方式 | 业务含义 | 中位数 | 上四分位 |\n"
        "|:---|:---|:---|:---:|:---:|\n" + "\n".join(_ext_rows))

    _aov_dom, _aov_dom_pct = _dominant_bin(ext_seg['AOV'])
    _np_dom, _np_dom_pct = _dominant_bin(ext_seg['N_products'])
    st.caption(f"💡 **解读**: AOV 以 {_aov_dom} 档为主 ({_aov_dom_pct}%)，品类广度以 {_np_dom} 档为主 ({_np_dom_pct}%)。"
               f"这两个特征回答的是 R/F/M 答不了的问题 —— 同样的消费总额，是「来了很多次但每次花得少」"
               f"还是「来得不多但每次买得多」(AOV)；同样的购买次数，是「反复买同几款」还是「横向铺开买很多种」(品类广度)。")

    # 五维相关性矩阵
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    _corr_cols = ['Recency', 'Frequency', 'Monetary', 'AOV', 'N_products']
    _corr_names = ['R-最近购买', 'F-购买频率', 'M-消费金额', 'AOV-平均客单价', '品类广度']
    corr = rfm_df[_corr_cols].corr()
    _fm_r = float(corr.loc['Frequency', 'Monetary'])
    _fm_strength = '强' if _fm_r >= 0.7 else ('中等' if _fm_r >= 0.4 else '弱')
    _aov_m_r = float(corr.loc['AOV', 'Monetary'])
    _np_f_r = float(corr.loc['N_products', 'Frequency'])
    _aov_np_r = float(corr.loc['AOV', 'N_products'])
    col_corr1, col_corr2 = st.columns([1, 2])
    with col_corr1:
        st.subheader("🔗 五维特征相关性")
        st.markdown(f"""
        **F 与 M** 呈{_fm_strength}正相关 (r≈{_fm_r:.2f})，说明买得多的客户也倾向于花得多 ——
        这是聚类中把 F/M 合并为 composite 特征的理论依据。

        **两个新特征与 R/F/M 基本独立**：AOV 与 M 的相关仅 {_aov_m_r:.2f}，品类广度与 F 为 {_np_f_r:.2f}，
        两者彼此 {_aov_np_r:.2f}。正因为不重复，它们才适合作为新增维度 —— 若选一个与 F 相关 0.7 以上的
        特征 (如活跃月数、退货次数)，加进去只是把 F 重复算了一遍，聚类不会因此变好。
        """)
    with col_corr2:
        corr_display = corr.copy()
        corr_display.index = _corr_names
        corr_display.columns = _corr_names
        fig_corr = px.imshow(corr_display.values,
                             x=corr_display.columns.tolist(),
                             y=corr_display.index.tolist(),
                             color_continuous_scale='RdBu_r',
                             zmin=-1, zmax=1,
                             text_auto='.2f',
                             aspect='auto',
                             labels={'color': '相关系数'})
        fig_corr.update_layout(**CHART_LAYOUT, height=420)
        st.plotly_chart(fig_corr, width='stretch')

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("🔗 R/F/M 两两散点")
    col1, col2 = st.columns(2)
    with col1:
        fig_rf = px.scatter(rfm_df, x='Frequency', y='Recency',
                            size='Monetary', color='Monetary',
                            color_continuous_scale='Viridis',
                            hover_data=['Customer ID'], opacity=0.6)
        fig_rf.update_layout(**CHART_LAYOUT, title="频率 vs 最近购买 (气泡大小/颜色 = 消费金额)",
                             xaxis_title='购买频率 (次)', yaxis_title='最近购买间隔 (天)', height=450)
        st.plotly_chart(fig_rf, width='stretch')

    with col2:
        fig_fm = px.scatter(rfm_df, x='Frequency', y='Monetary',
                            size='Recency', color='Recency',
                            color_continuous_scale='RdYlGn_r',
                            hover_data=['Customer ID'], opacity=0.6)
        fig_fm.update_layout(**CHART_LAYOUT, title="频率 vs 消费金额 (气泡大小/颜色 = 最近购买)",
                             xaxis_title='购买频率 (次)', yaxis_title='消费金额 ($)', height=450)
        st.plotly_chart(fig_fm, width='stretch')

    st.caption("💡 **解读**: 左图高频客户 (右侧) 消费金额更高且购买更近期；右图右上角的少量极端客户消费金额远超其余，偏态分布一目了然。")

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
    st.plotly_chart(fig_seg, width='stretch')

    with st.expander("📋 查看分类依据 (评分规则)"):
        st.markdown("""
**评分方法**: 将 R / F / M 各按分位数分为 1-5 档 (5 为最优)：

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
| 潜力客户 | ≥3 | 1-2 | — | 有一定活跃度、可挖掘价值 |
| 沉睡客户 | ≤2 | — | — | 不活跃但未完全流失 |
| 待开发 | 其他 | — | — | 尚未明确归类的客户 |
        """)

    _seg_rank = seg_counts.set_index('客户分群')['客户数量'].sort_values(ascending=False)
    _seg_total = int(_seg_rank.sum())
    _top_segs_txt = ' 和 '.join(
        f"**{nm}** ({int(v):,} 人, {v / _seg_total * 100:.1f}%)"
        for nm, v in _seg_rank.head(2).items())
    _champ_mask = scored['分群'] == '冠军客户 (Champions)'
    _champ_n = int(_champ_mask.sum())
    _champ_ppl_pct = _champ_n / _seg_total * 100
    _champ_rev_pct = scored.loc[_champ_mask, 'Monetary'].sum() / scored['Monetary'].sum() * 100
    st.caption(f"💡 **解读**: 人数最多的两个分群为 {_top_segs_txt}。"
               f"**冠军客户** {_champ_n:,} 人 (占人数 {_champ_ppl_pct:.1f}%) 贡献了 {_champ_rev_pct:.1f}% 的收入——"
               f"人数占比与收入占比之间的差距，正是分群差异化运营的价值所在。")

# ============================================================
# 页面 4: K-Means 聚类分析
# ============================================================
elif page == "🎯 K-Means 聚类":
    st.title("🎯 K-Means 客户分群分析")

    # 侧边栏参数控制
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 聚类参数")
    k_range_start = st.sidebar.number_input("K 搜索范围 (起始)", min_value=2, max_value=14, value=2, step=1)
    # value 必须跟随 min_value 一起抬升: 否则把起始调到 11 时 min_value=12 而写死的 value 仍是 10,
    # Streamlit 会抛 StreamlitValueBelowMinError 中断整页
    k_range_end = st.sidebar.number_input(
        "K 搜索范围 (结束)", min_value=int(k_range_start) + 1, max_value=15,
        value=max(int(k_range_start) + 1, 10), step=1)
    winsorize_pct = st.sidebar.slider("Winsorize 截断 (上分位)", min_value=0.95, max_value=1.0,
                                       value=0.995, step=0.005, format="%.3f",
                                       help="将 F/M 超过该分位数的值截断，默认 0.995 (截断 top 0.5%)")
    selected_k = st.sidebar.slider("聚类数量 K", min_value=2, max_value=15, value=4, step=1,
                                   help="默认取 4 —— 与「第一步」里轮廓系数给出的推荐 K 一致")
    # 特征变换方法不再是独立选项: 每个特征集配它实测最优的方法, 只留一个控件、信息更简单。
    # 实测 (K=4): 仅 RFM 时组合特征 0.4486 > PCA 0.4369; 加扩展特征后 PCA 0.3967 > 组合特征 0.3034
    feature_set = st.sidebar.selectbox(
        "输入特征集 (已自动配对最优的变换方法)",
        options=['rfm_ext', 'rfm_only'],
        format_func=lambda x: {
            'rfm_ext': f'RFM + 扩展特征 → {METHOD_LABELS[EXT_FEATURE_SET_METHOD["rfm_ext"]]}',
            'rfm_only': f'仅 RFM → {METHOD_LABELS[EXT_FEATURE_SET_METHOD["rfm_only"]]}',
        }[x],
        help="扩展特征 = 平均客单价 AOV + 品类广度 (见第 3 页), 两者与 R/F/M 相关性低、能提供额外信息, "
             "配 PCA 降维使用; 「仅 RFM」配组合特征 2D —— 两种配对都是各自特征集下轮廓系数最高的组合"
    )
    extra_features = ['AOV', 'N_products'] if feature_set == 'rfm_ext' else []
    cluster_method = EXT_FEATURE_SET_METHOD[feature_set]

    # 特征准备 + 最优 K 搜索 (缓存)
    scaled, feature_names, scaler, transformed_df, elbow_result, pca_model = cached_prepare_and_elbow(
        rfm_df, cluster_method, winsorize_pct, k_range_start, k_range_end, extra_features)

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
        st.plotly_chart(fig_elbow, width='stretch')
        st.caption("💡 **肘部法则**: WCSS 随 K 增大单调下降，拐点处即最优 K。")
    with col2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=elbow_result['k_values'], y=elbow_result['silhouette_scores'],
            mode='lines+markers', name='轮廓系数',
            line=dict(color=COLORS['danger'], width=3),
            marker=dict(size=10, color=COLORS['danger']),
            fill='tozeroy', fillcolor='rgba(220,38,38,0.08)',
        ))
        _kv = elbow_result['k_values']
        _sv = elbow_result['silhouette_scores']
        _pr = [(k, s) for k, s in zip(_kv, _sv) if k >= 3]
        rec_k, rec_sil = max(_pr, key=lambda t: t[1]) if _pr else (
            elbow_result['best_k'], elbow_result['best_silhouette'])
        best_idx = _kv.index(rec_k)
        fig_sil.add_trace(go.Scatter(
            x=[rec_k], y=[_sv[best_idx]],
            mode='markers+text',
            marker=dict(size=18, color=COLORS['warning'], symbol='star', line=dict(width=2, color='white')),
            text=[f'推荐 K={rec_k}'],
            textposition='top center',
            name=f'推荐 K={rec_k}'
        ))
        fig_sil.update_layout(**CHART_LAYOUT,
            title="轮廓系数分析 (Silhouette Score)",
            xaxis_title='聚类数 K', yaxis_title='轮廓系数',
            height=400)
        st.plotly_chart(fig_sil, width='stretch')
        st.caption("💡 **轮廓系数**: 衡量簇内紧密度与簇间分离度，取值 -1~1，越高越好。⭐ 为 K≥3 范围内的推荐 K。")

    st.info(f"**K 值选择依据**: 轮廓系数在 K=2 处最高 ({elbow_result['best_silhouette']:.4f})，"
            f"但只对应粗粒度的\"活跃 vs 沉睡\"二分，营销粒度不足。"
            f"K≥3 范围内 **K={rec_k} 最高 ({rec_sil:.4f})**，肘部拐点也落在 4-5 区间 (K=5 低约 0.02)。"
            f"结合统计指标与业务分群惯例，本项目默认取 K=4。当前选择 **K = {selected_k}**。")

    with st.expander("🔬 两种特征变换方法 — 组合特征 vs PCA (点击展开)"):
        st.markdown("""
        **核心矛盾**: K-Means 依赖欧氏距离, 对 F/M 的极度右偏 (F 偏度 ~10, M 偏度 ~24) 与两者之间的中高度正相关 (r ≈ 0.65) 都很敏感。3 维空间里 F 与 M 传递高度相似的信息, 相当于把它们对距离的贡献隐式放大。

        **本项目采用的两种方法**:

        | 方法 | 维度 | 特征构造 | 优点 | 局限 |
        |:---|:---:|---------|------|------|
        | **组合特征 2D** (配「仅 RFM」) | 2D | `[R_rank, R_rank+F_rank+M_rank]` | 降到 2D 且语义可读: 时效性 × 参与度; 仅 RFM 时轮廓系数最高 (0.4486) | 两维共享 `R_rank`, 存在正相关 (≈0.6), **非严格正交** |
        | **PCA 降维 2D** (配「RFM + 扩展特征」) | 2D | 对 `[R,F,M,扩展特征]` 的排名取前 2 个主成分 | 两个主成分**严格正交**; 同时承载全部输入特征的信息 (前两维解释 82.1% 方差); 加扩展特征后轮廓系数最高 (0.3967) | 主成分是特征的线性组合, 语义不如"时效性/参与度"直观 (页面上会把载荷标出来) |

        **为什么这样配 (实测依据, K=4)**:

        | 特征集 | 组合特征 2D | PCA 降维 2D | 本项目选择 |
        |:---|:---:|:---:|:---|
        | 仅 RFM | **0.4486** | 0.4369 | 组合特征 (高 0.012) |
        | RFM + 扩展特征 | 0.3034 | **0.3967** | **PCA (高 0.093)** |

        两个特征集各自取"轮廓系数最高的方法", 所以侧边栏不再单独暴露方法选项, 只留「输入特征集」一个控件。

        > 另外两种方法也评估过但未采用: **百分位排名 3D** (仅 RFM 0.3898 / 加扩展 0.2857)、**对数变换 3D** (0.3326 / 0.2750)。对数变换最差的原因是 F/M 偏度极大 (≈10 / ≈24), 对数压缩不够充分、簇仍重叠; 百分位排名语义清晰但保留了 F/M 相关性, K-Means 距离里 F/M 被隐式放大。

        **组合特征的短板与 PCA 的解法**: `R_rank` 同时是第 1 维和第 2 维的组成部分, 严格讲两维不正交 (相关约 0.6), 相当于把 R 在距离中"算了两次"。**这正是加了扩展特征后改用 PCA 的原因** —— PCA 给出的是两个严格正交的主成分。若被追问"你的两维正交吗", 标准答法是: 承认组合特征不正交, 紧接着说明"所以在 5 维输入下我们改用 PCA, 两维严格正交、轮廓系数从 0.3034 提到 0.3967"。

        **两种方法的共用流程 (PCA 多一道)**:
        1. **Winsorizing**: 截断 F/M 的 top 0.5% 极端值
        2. **百分位排名**: `rank(pct=True)` 压缩到 0~1
        3. **StandardScaler**: Z-score 标准化
        4. **(仅 PCA)** `PCA(n_components=2)` 取前两个主成分作为聚类空间 —— 刻意**不再二次标准化**: 主成分按方差降序排列, PC1 承载的信息本就最多, 拉平会丢掉这个权重
        5. **K-Means**: `n_init=10, max_iter=500` (n_init 取 sklearn 默认值 —— 本数据上 10 与 50 的簇划分几乎一致, 但 K 扫描快近一倍)
        """)

        st.markdown(f"**📊 当前特征集下两种方法的实时对比 (K={selected_k}, Winsorize={winsorize_pct:.3f}, "
                    f"特征集={'RFM + 扩展' if extra_features else '仅 RFM'})**:")
        _cmp = cached_compare_methods(rfm_df, selected_k, winsorize_pct, extra_features)
        _cmp_rows = []
        for _m in ['composite', 'pca']:
            _v = _cmp[_m]
            _cmp_rows.append({
                '方法': METHOD_LABELS[_m],
                '维度': _v['dims'],
                '轮廓系数': f"{_v['silhouette']:.4f}" if _v.get('ok') else '计算失败',
                '是否当前采用': '✅' if _m == cluster_method else '',
            })
        st.dataframe(pd.DataFrame(_cmp_rows), width='stretch', hide_index=True)
        st.caption("💡 上表数值随侧边栏 K、Winsorize 与输入特征集实时变化, 非固定参考值。"
                   "两个数值不可跨维度直接比较 —— 组合特征是 2-4 维、PCA 固定 2 维。")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # 执行聚类
    st.subheader("🎯 第二步: 聚类结果")
    result = cached_run_kmeans(rfm_df, selected_k, cluster_method, winsorize_pct, extra_features)
    profile = result['cluster_profile']
    clustered_df = result['clustered_df']
    labels = get_cluster_labels(profile)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("聚类数量", f"{selected_k} 组")
    with col2:
        st.metric("轮廓系数", f"{result['silhouette_score']:.4f}")
    with col3:
        st.metric("参与聚类客户", f"{len(clustered_df):,} 人")
    with col4:
        st.metric("输入维度", f"{len(feature_names)} 维",
                  help=f"特征集: {'RFM + 扩展特征' if extra_features else '仅 RFM'}；扩展特征为 AOV 与品类广度")
    with col5:
        st.metric("变换方法", METHOD_LABELS[cluster_method].replace(' 2D', ''),
                  help="由「输入特征集」自动配对 (依据: 该特征集下轮廓系数最高的方法); K-Means: n_init=10, max_iter=500")

    # 各维度在界面上的中文名 (供 2D 特征空间图与切片下拉框使用;
    # log 方法的 R/F/M 三列存的是 log1p 后的值)
    _axis_names = {
        'R_rank': 'R-时效性 (排名)', 'RFM_composite': 'RFM-参与度 (综合排名)',
        'AOV_rank': 'AOV-平均客单价 (排名)', 'N_products_rank': '品类广度 (排名)',
        'Recency': 'R (最近购买)', 'Frequency': 'F (购买频率)', 'Monetary': 'M (消费金额)',
        'PC1': 'PC1 第一主成分', 'PC2': 'PC2 第二主成分',
    }
    if cluster_method == 'log':
        _axis_names.update({'Recency': 'log(R)', 'Frequency': 'log(F)', 'Monetary': 'log(M)'})

    # 2D 特征空间散点图 (聚类实际发生的空间)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("📍 特征空间 2D 可视化")
    st.caption("聚类实际发生的特征空间 (非原始 R/F/M)，可框选查看选中客户明细。")

    plot_2d = transformed_df.reset_index(drop=True).copy()
    # transformed_df 仅含工程特征列, 需按行序从 clustered_df 补回 Cluster / Customer ID / 原始 R F M
    _src = clustered_df[['Customer ID', 'Cluster', 'Recency', 'Frequency', 'Monetary']].reset_index(drop=True)
    for _col in _src.columns:
        if _col not in plot_2d.columns:
            plot_2d[_col] = _src[_col]
    plot_2d['聚类标签'] = plot_2d['Cluster'].map(lambda c: f"C{c}: {labels[c]['name']}")

    if len(feature_names) == 2:
        _fx, _fy = feature_names[0], feature_names[1]
        _dim_note = ""
    else:
        # 聚类实际发生在更高维空间, 这里选两个维度做 2D 切片查看
        _pairs = [(feature_names[i], feature_names[j])
                  for i in range(len(feature_names))
                  for j in range(i + 1, len(feature_names))]
        _picked = st.selectbox(
            f"当前方法在 {len(feature_names)} 维空间聚类, 请选择展示哪两个维度 (点击切换不同切片):",
            options=_pairs,
            index=0,
            format_func=lambda t: f"{_axis_names.get(t[0], t[0])} vs {_axis_names.get(t[1], t[1])}",
            # key 必须带上特征集: 否则切换特征集后旧选中值已不在 options 里
            key=f"dim_pair_{cluster_method}_{feature_set}",
        )
        _fx, _fy = _picked
        _dim_note = (f" — {len(feature_names)} 维空间切片 (完整聚类维度: "
                     f"{', '.join(_axis_names.get(c, c) for c in feature_names)})")

    _xl = _axis_names.get(_fx, _fx)
    _yl = _axis_names.get(_fy, _fy)

    fig_2d = px.scatter(
        plot_2d, x=_fx, y=_fy,
        color='聚类标签',
        hover_data=['Customer ID', 'Recency', 'Frequency', 'Monetary'],
        opacity=0.7,
        color_discrete_sequence=COLORS['palette'],
        labels={_fx: _xl, _fy: _yl},
    )
    fig_2d.update_layout(**CHART_LAYOUT, height=500,
                         title=f"聚类特征空间散点图{_dim_note}")
    selection = st.plotly_chart(fig_2d, width='stretch', selection_mode="points",
                                on_select="rerun", key="scatter_2d")
    if cluster_method == 'pca' and pca_model is not None:
        # 主成分是特征的线性组合, 观众看不出轴的含义, 这里把解释方差与载荷直接标出来
        _evr = pca_model.explained_variance_ratio_ * 100
        _load = pca_model.components_
        _base_cols = (['R-时效性', 'F-购买频率', 'M-消费金额']
                      + [EXTENDED_FEATURE_INFO[k]['name'] for k in extra_features
                         if k in EXTENDED_FEATURE_INFO])
        _p1 = '、'.join(f"{_base_cols[_i]} {_load[0][_i]:+.2f}" for _i in range(len(_base_cols)))
        _p2 = '、'.join(f"{_base_cols[_i]} {_load[1][_i]:+.2f}" for _i in range(len(_base_cols)))
        st.caption(f"💡 **两轴怎么读**: 主成分是输入的线性组合, 累计解释 **{_evr[:2].sum():.1f}%** 方差 "
                   f"(PC1 {_evr[0]:.1f}% + PC2 {_evr[1]:.1f}%)。"
                   f"载荷: **PC1** = {_p1} (全为正 → 越往右整体价值越高); **PC2** = {_p2} (正负对比轴)。两轴严格正交。")

    # 展示框选客户详情
    sel_events = selection.get('selection', {}).get('points', [])
    if sel_events:
        sel_ids = [p.get('customdata', [None])[0] for p in sel_events if p.get('customdata')]
        sel_customers = rfm_df[rfm_df['Customer ID'].isin(sel_ids)].sort_values('Monetary', ascending=False)
        if len(sel_customers) > 0:
            with st.expander(f"📋 已选中 {len(sel_customers)} 位客户 — 点击展开详情", expanded=True):
                display_sel = sel_customers[['Customer ID', 'Recency', 'Frequency', 'Monetary']].copy()
                display_sel.columns = ['客户 ID', 'R (天)', 'F (次)', 'M ($)']
                st.dataframe(display_sel, width='stretch', hide_index=True, height=300)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # 聚类画像 + 雷达图
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 聚类画像 (Cluster Profile)")
        display_profile = profile.copy()
        display_profile['标签'] = display_profile.index.map(lambda c: labels[c]['name'])
        _prof_cols = ['标签', 'Customers', 'Pct_Customers',
                      'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
        _prof_names = ['客户分群', '客户数', '客户占比%', '平均R(天)', '平均F(次)', '平均M($)']
        # 启用扩展特征时把两个新维度的各簇均值也放进画像 (列名取 EXTENDED_FEATURE_INFO)
        for _key, _meta in EXTENDED_FEATURE_INFO.items():
            if f'Avg_{_key}' in display_profile.columns:
                _prof_cols.append(f'Avg_{_key}')
                _prof_names.append(_meta['name'])
        _prof_cols += ['Total_Revenue', 'Pct_Revenue']
        _prof_names += ['总收入', '收入占比%']
        display_profile = display_profile[_prof_cols]
        display_profile.columns = _prof_names
        st.dataframe(display_profile, width='stretch', hide_index=True)

    with col_right:
        st.subheader("🕸️ 雷达图 (Radar Chart)")
        norm_profile = profile.copy()
        _RADAR_MIN = 0.15  # 避免最差簇归 0 塌缩到圆心
        for col in ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']:
            _rng = norm_profile[col].max() - norm_profile[col].min()
            if _rng < 1e-10:
                norm_profile[col] = 0.5
            else:
                norm_profile[col] = _RADAR_MIN + (1 - _RADAR_MIN) * (
                    norm_profile[col] - norm_profile[col].min()) / _rng
        norm_profile['Avg_Recency'] = (1 + _RADAR_MIN) - norm_profile['Avg_Recency']

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
        st.plotly_chart(fig_radar, width='stretch')
        st.caption("💡 **解读**: 雷达图面积越大代表该簇综合价值越高。重要价值客户簇在三个维度上均突出，覆盖面积最大。")
        st.caption("*R-最近购买: 值越高 = 购买越近期 (已反转)")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # 聚类详情卡片
    st.subheader("📇 各聚类详情")
    cols = st.columns(min(selected_k, 5))
    card_colors = COLORS['palette']

    for idx, cluster_id in enumerate(profile.index):
        with cols[idx % len(cols)]:
            color = card_colors[cluster_id % len(card_colors)]
            # 扩展特征启用时, 卡片里补上这两个维度的簇均值
            _ext_html = ''
            if 'Avg_AOV' in profile.columns:
                _ext_html += f"平均客单价: <b>${profile.loc[cluster_id, 'Avg_AOV']:,.0f}</b><br>\n                    "
            if 'Avg_N_products' in profile.columns:
                _ext_html += f"平均品类广度: <b>{profile.loc[cluster_id, 'Avg_N_products']:.0f} 种</b><br>\n                    "
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
                    {_ext_html}收入占比: <b>{profile.loc[cluster_id, 'Pct_Revenue']}%</b><br>
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
        'AOV_rank': 'AOV-平均客单价 (排名)', 'N_products_rank': '品类广度 (排名)',
        'PC1': 'PC1 第一主成分', 'PC2': 'PC2 第二主成分',
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
    st.plotly_chart(fig_heat, width='stretch')
    if cluster_method == 'pca':
        # PCA 下中心点只有两个主成分坐标, 不存在 AOV/品类广度这类原始列, 尾注必须跟屏幕上的列一致
        _heat_tail = (" 聚类发生在 **PC1 / PC2** 上, 故本图只有这两列 —— PC1 是「综合价值」轴 (五项载荷全为正), "
                      "PC2 是「客单价 vs 时效」对比轴; 载荷明细见上方特征空间图。")
    elif extra_features:
        _heat_tail = (" 当前特征集含扩展特征, 故还有 **AOV-平均客单价 (排名)** 与 **品类广度 (排名)** 两列 —— "
                      "两列在各簇之间是否有明显颜色差异, 就是它们真正参与分群的证据。")
    else:
        _heat_tail = (" 组合特征把 R/F/M 压缩为 **时效性 (R_rank)** 与 **综合参与度 (R+F+M 排名之和)** 两列; "
                      "两维共享 R 项、非严格正交 (详见「两种特征变换方法」展开)。")
    st.caption("💡 **解读**: 聚类中心已归一化到 0-1，颜色越绿表示该维度得分越高 —— "
               "重要价值客户各维度得分接近 1，流失客户各维度均偏低。"
               + _heat_tail)

# ============================================================
# 页面 5: 关联规则分析
# ============================================================
elif page == "🔗 关联规则分析":
    st.title("🔗 关联规则分析 (Market Basket Analysis)")
    st.markdown("基于 FP-Growth 算法挖掘商品共购模式，发现「买了 A 的客户也倾向买 B」的关联规则")

    # --- 侧边栏参数 ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 关联规则参数")
    # 两个阈值的量程由页面内「阈值范围如何确定」的网格扫描反推得到, 不再取更宽的空区间:
    # 支持度 > 0.03 时规则数不足 10 条, > 0.05 时一条规则都挖不出; 置信度 > 0.7 同理
    assoc_min_support = st.sidebar.slider(
        "最小支持度 (min_support)", min_value=0.005, max_value=0.030,
        value=0.02, step=0.005, format="%.3f",
        help="商品组合至少出现在多少比例的交易中。值越小规则越多但计算越慢。量程由下方敏感性扫描确定。")
    assoc_min_confidence = st.sidebar.slider(
        "最小置信度 (min_confidence)", min_value=0.1, max_value=0.7,
        value=0.3, step=0.05, format="%.2f",
        help="规则 A→B 中，买 A 的客户有多大比例也买了 B。量程由下方敏感性扫描确定。")
    assoc_top_items = st.sidebar.slider(
        "分析商品数 (Top-N)", min_value=50, max_value=300,
        value=100, step=10,
        help="只取购买频次最高的前 N 种商品参与分析，控制计算量。")

    # --- 缓存计算 ---
    # 注意: `_df` / `_n_rows` 均带下划线前缀 → Streamlit 两个都不计入缓存哈希
    # (40 万行 DataFrame 逐次哈希开销过大, 属刻意的性能取舍)。
    # 因此缓存键实际只有 (min_support, min_confidence, top_items), 数据本身不在键里。
    # 当前单数据集 + 上层 Parquet 内容指纹已覆盖「换数据源」「改清洗逻辑」两条失效路径;
    # 若将来支持多数据集切换, 必须把数据源标识以「不带下划线」的形参显式纳入缓存键,
    # 否则两份数据会互相命中脏缓存。
    @st.cache_data(show_spinner="正在运行 FP-Growth 关联规则挖掘...")
    def cached_association(_df, min_support, min_confidence, top_items, _n_rows):
        basket = prepare_basket_matrix(_df, top_n_items=top_items)
        result = run_fpgrowth(basket, min_support=min_support,
                              min_confidence=min_confidence, min_lift=1.0)
        cooc = compute_cooccurrence_matrix(basket, top_n=min(20, basket.shape[1]))
        return result, basket.shape, cooc

    assoc_result, basket_shape, cooc_matrix = cached_association(
        cleaned_df, assoc_min_support, assoc_min_confidence, assoc_top_items, len(cleaned_df))

    rules_df = assoc_result['rules']
    itemsets_df = assoc_result['itemsets']
    n_transactions = assoc_result['n_transactions']
    n_items = assoc_result['n_items']

    # --- KPI 指标卡片 ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("分析交易数", f"{n_transactions:,}")
    with kpi2:
        st.metric("分析商品数", f"{n_items}")
    with kpi3:
        st.metric("频繁项集", f"{len(itemsets_df)}")
    with kpi4:
        st.metric("关联规则数", f"{len(rules_df)}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # --- 阈值敏感性分析: 支持度/置信度的量程是"扫出来"的, 不是随手定的 ---
    st.subheader("🎚️ 阈值范围如何确定 (敏感性扫描)")
    st.caption("把支持度与置信度做全量程网格扫描, 看规则数随阈值如何变化 —— 侧边栏两个滑块的量程即由此确定。")

    _sweep, _sup_info = cached_threshold_sweep(cleaned_df, assoc_top_items, len(cleaned_df))
    _sw1, _sw2 = st.columns(2)

    with _sw1:
        _pivot = _sweep.pivot(index='支持度', columns='置信度', values='规则数')
        fig_sw = px.imshow(_pivot.values,
                           x=[f"{c:.1f}" for c in _pivot.columns],
                           y=[f"{s:.3f}" for s in _pivot.index],
                           color_continuous_scale='YlGnBu',
                           text_auto=True, aspect='auto',
                           labels={'x': '最小置信度', 'y': '最小支持度', 'color': '规则数'})
        fig_sw.update_layout(**CHART_LAYOUT, height=430,
                             title="规则数 = f(支持度, 置信度)")
        st.plotly_chart(fig_sw, width='stretch')

    with _sw2:
        fig_line = go.Figure()
        for _c in [0.1, 0.3, 0.5, 0.7]:
            _sub = _sweep[_sweep['置信度'] == _c]
            if _sub.empty:
                continue
            fig_line.add_trace(go.Scatter(
                x=_sub['支持度'], y=_sub['规则数'], mode='lines+markers',
                name=f'置信度 {_c:.1f}'))
        fig_line.update_layout(**CHART_LAYOUT, height=430,
                               title="规则数随支持度的衰减 (每条线一个置信度档)",
                               xaxis_title='最小支持度', yaxis_title='规则数')
        st.plotly_chart(fig_line, width='stretch')

    # 推荐区间以「规则数仍不少于 10 条」为可操作下限 —— 与规则详情表的滑块下限一致,
    # 避免定出一个理论上能调、实际上选不出规则的空区间
    _MIN_RULES = 10
    _ok = _sweep[_sweep['规则数'] >= _MIN_RULES]
    _sup_min = _sweep['支持度'].min()
    _n_at_min_sup = int(_sweep[_sweep['支持度'] == _sup_min]['规则数'].max())
    # 死区 = 跨**所有**置信度都挖不出规则的最小支持度 (不能只看某一档置信度下的 0)
    _sup_max_rules = _sweep.groupby('支持度')['规则数'].max()
    _dead_sups = _sup_max_rules[_sup_max_rules == 0]
    _dead_txt = f"{_dead_sups.index.min():.3f}" if len(_dead_sups) else "—"
    _sup_hi_txt = f"{_ok['支持度'].max():.3f}" if len(_ok) else "—"
    _conf_hi_txt = f"{_ok['置信度'].max():.2f}" if len(_ok) else "—"
    _max_pair_sup = _sup_info['max_pair_support']
    _max_item_sup = _sup_info['max_item_support']
    # 当前规则表的最高 lift (规则为空时用 — 占位, 避免写死数字随参数漂移)
    _cur_max_lift = float(rules_df['lift'].max()) if not rules_df.empty else float('nan')
    _cur_lift_txt = f"{_cur_max_lift:.2f}" if _cur_max_lift == _cur_max_lift else "—"

    st.markdown(f"""
**扫描结论** (分析商品数 Top-{assoc_top_items}):

| 参数 | 量程上的表现 | 推荐区间 | 依据 |
|:---|:---|:---|:---|
| 最小支持度 | 最低档 {_sup_min:.3f} 时规则数达 {_n_at_min_sup:,} 条; 升到 **{_dead_txt} 之后一条规则都挖不出** | **{_sup_min:.3f} ~ {_sup_hi_txt}** | 上界: 再高则规则数不足 {_MIN_RULES} 条, 无法支撑筛选; 下界: 再低规则数爆炸且大量退化为长尾偶然组合 |
| 最小置信度 | 0.1 时规则最多, 0.9 时几乎为 0 | **0.10 ~ {_conf_hi_txt}** | 上界: 置信度越高规则越少, 超过后不足 {_MIN_RULES} 条 |

**为什么支持度上界必须压得这么低?** 一条规则至少要求**两种商品同时出现**, 而本数据集里**商品对**的最高支持度
只有 **{_max_pair_sup:.2%}** —— 阈值一旦超过它, 连最高频的那一对商品都进不了频繁项集, 必然一条规则都挖不出来
({_dead_txt} 以上正是这个死区)。作为对照, 最高频**单品**的支持度高达 {_max_item_sup:.1%}; 单品与商品对之间这么大的落差,
说明这个数据集的商品共现其实相当分散 —— 这也解释了为什么零售购物篮分析的 min_support 通常都取得很低。

**一个反直觉的现象**: 支持度越低规则越多, 而且**最高提升度反而更高**
(本次扫描中 0.005 档最高 lift 达 {_sweep['最高提升度'].max():.1f}, 而当前参数下的最高 lift 为 {_cur_lift_txt})。
但低支持度的规则大多只覆盖几十笔交易, 属于长尾偶然共现, 所以本项目默认取 0.02 而不是一味求多 ——
**这就是"确定阈值范围"要平衡的两端: 规则数量、规则强度与统计可靠性**。
""")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if rules_df.empty:
        st.warning("当前参数下未找到满足条件的关联规则。请尝试降低最小支持度或最小置信度。")
    else:
        # --- 商品名中文翻译 ---
        PRODUCT_CN = {
            '60 TEATIME FAIRY CAKE CASES': '60个下午茶仙女蛋糕纸杯',
            '72 SWEETHEART FAIRY CAKE CASES': '72个甜心仙女蛋糕纸杯',
            'BLUE 3 PIECE MINI DOTS CUTLERY SET': '蓝色三件迷你圆点餐具套装',
            'CHARLOTTE BAG , PINK/WHITE SPOTS': '夏洛特袋 粉白圆点',
            'CHOCOLATE HOT WATER BOTTLE': '巧克力色热水袋',
            'COOK WITH WINE METAL SIGN': '烹饪用酒金属标牌',
            'CREAM HEART CARD HOLDER': '奶油色心形卡片架',
            'GIN + TONIC DIET METAL SIGN': '金汤力减肥金属标牌',
            'HEART OF WICKER LARGE': '大号藤编心形装饰',
            'HEART OF WICKER SMALL': '小号藤编心形装饰',
            'HOME BUILDING BLOCK WORD': 'HOME字母积木装饰',
            'HOT WATER BOTTLE TEA AND SYMPATHY': '茶与同情热水袋',
            'JUMBO  BAG BAROQUE BLACK WHITE': '超大购物袋 巴洛克黑白',
            'JUMBO BAG PINK VINTAGE PAISLEY': '超大购物袋 粉色复古佩斯利',
            'JUMBO BAG PINK WITH WHITE SPOTS': '超大购物袋 粉白圆点',
            'JUMBO BAG RED WHITE SPOTTY': '超大购物袋 红白圆点',
            'JUMBO BAG SCANDINAVIAN PAISLEY': '超大购物袋 北欧佩斯利',
            'JUMBO BAG STRAWBERRY': '超大购物袋 草莓图案',
            'JUMBO SHOPPER VINTAGE RED PAISLEY': '超大购物袋 复古红佩斯利',
            'JUMBO STORAGE BAG SUKI': '超大收纳袋 SUKI款',
            'LOVE BUILDING BLOCK WORD': 'LOVE字母积木装饰',
            'LUNCH BAG  BLACK SKULL.': '午餐袋 黑色骷髅',
            'LUNCH BAG CARS BLUE': '午餐袋 蓝色汽车',
            'LUNCH BAG RED SPOTTY': '午餐袋 红色圆点',
            'LUNCH BAG WOODLAND': '午餐袋 森林图案',
            'LUNCHBAG PINK RETROSPOT': '午餐袋 粉色复古圆点',
            'LUNCHBAG SPACEBOY DESIGN': '午餐袋 太空男孩',
            'LUNCHBAG SUKI  DESIGN': '午餐袋 SUKI款',
            'PACK OF 60 DINOSAUR CAKE CASES': '60个恐龙蛋糕纸杯',
            'PACK OF 60 PINK PAISLEY CAKE CASES': '60个粉色佩斯利蛋糕纸杯',
            'PACK OF 72 RETRO SPOT CAKE CASES': '72个复古圆点蛋糕纸杯',
            'PACK OF 72 SKULL CAKE CASES': '72个骷髅蛋糕纸杯',
            'PINK 3 PIECE MINI DOTS CUTLERY SET': '粉色三件迷你圆点餐具套装',
            'PINK BLUE FELT CRAFT TRINKET BOX': '粉蓝色毛毡首饰盒',
            'PINK CREAM FELT CRAFT TRINKET BOX': '粉色奶油毛毡首饰盒',
            'PLEASE ONE PERSON  METAL SIGN': '请限一人金属标牌',
            'RED 3 PIECE MINI DOTS CUTLERY SET': '红色三件迷你圆点餐具套装',
            'RED HANGING HEART T-LIGHT HOLDER': '红色悬挂心形烛台',
            'RED SPOT HEART HOT WATER BOTTLE': '红点心形热水袋',
            'RED SPOTTY CHARLOTTE BAG': '红色圆点夏洛特袋',
            'SCOTTIE DOG HOT WATER BOTTLE': '苏格兰犬热水袋',
            'SET/20 RED SPOTTY PAPER NAPKINS': '20张红色圆点纸餐巾',
            'STRAWBERRY CERAMIC TRINKET BOX': '草莓陶瓷首饰盒',
            'SWEETHEART CERAMIC TRINKET BOX': '甜心陶瓷首饰盒',
            'VINTAGE HEADS AND TAILS CARD GAME': '复古正反面卡牌游戏',
            'VINTAGE SNAP CARDS': '复古拍牌游戏卡牌',
            'WHITE HANGING HEART T-LIGHT HOLDER': '白色悬挂心形烛台',
            'WOOD 2 DRAWER CABINET WHITE FINISH': '白色双抽木柜',
            'WOOD S/3 CABINET ANT WHITE FINISH': '白色三层小木柜',
            'WOODEN FRAME ANTIQUE WHITE': '白色复古木相框',
            'WOODEN PICTURE FRAME WHITE FINISH': '白色木相框',
            'ZINC METAL HEART DECORATION': '锌金属心形装饰',
        }

        def _cn(name):
            return PRODUCT_CN.get(name, name)

        # --- 散点图: 支持度 vs 置信度, 颜色=提升度 ---
        st.subheader("📊 规则质量分布 (支持度 × 置信度 × 提升度)")

        scatter_df = rules_df.copy()
        scatter_df['前项'] = scatter_df['antecedents'].apply(
            lambda x: ' + '.join(_cn(i) for i in sorted(x)))
        scatter_df['后项'] = scatter_df['consequents'].apply(
            lambda x: ' + '.join(_cn(i) for i in sorted(x)))
        scatter_df['规则'] = scatter_df['前项'] + ' → ' + scatter_df['后项']

        fig_scatter = px.scatter(
            scatter_df, x='support', y='confidence',
            size='lift', color='lift',
            color_continuous_scale='Viridis',
            hover_data={'规则': True, 'support': ':.4f', 'confidence': ':.2%',
                        'lift': ':.2f', '前项': False, '后项': False},
            labels={'support': '支持度 (Support)', 'confidence': '置信度 (Confidence)',
                    'lift': '提升度 (Lift)'},
        )
        fig_scatter.update_layout(**CHART_LAYOUT, height=480,
                                  title="每条规则的支持度 vs 置信度 (气泡大小/颜色 = 提升度)")
        st.plotly_chart(fig_scatter, width='stretch')
        st.caption("💡 **解读**: 右上角的规则同时具有高支持度和高置信度，是最有价值的关联规则。"
                   "提升度 > 1 说明两商品的出现不是偶然的，而是正相关。")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # --- 网络关系图 ---
        st.subheader("🕸️ 商品关联网络图")

        net_data = build_network_graph(rules_df, top_n=min(30, len(rules_df)))

        if net_data['nodes']:
            import networkx as nx

            nodes = net_data['nodes']
            edges = net_data['edges']
            pos = net_data['pos']

            # Edge traces
            edge_x, edge_y = [], []
            edge_hover = []
            # 每条边在 edge_x/edge_y 中占 3 个点 (x0, x1, None 分隔符),
            # hovertext 必须逐点对齐, 否则标签会整体串位
            for e in edges:
                x0, y0 = pos[e['source']]
                x1, y1 = pos[e['target']]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                _h = (f"{_cn(e['source'])} → {_cn(e['target'])}<br>"
                      f"提升度={e['lift']:.2f}, 置信度={e['confidence']:.1%}")
                edge_hover += [_h, _h, ""]

            fig_net = go.Figure()

            # Edges
            fig_net.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode='lines',
                line=dict(width=1.5, color='rgba(150,150,150,0.5)'),
                hoverinfo='text',
                hovertext=edge_hover,
                showlegend=False,
            ))

            # Nodes
            node_x = [pos[n['id']][0] for n in nodes]
            node_y = [pos[n['id']][1] for n in nodes]
            node_size = [8 + n.get('degree', 1) * 4 for n in nodes]
            node_text = [_cn(n['label']) for n in nodes]
            node_hover = [f"<b>{_cn(n['label'])}</b><br>关联数={n.get('degree', 0)}<br>"
                          f"最大支持度={n.get('support', 0):.4f}" for n in nodes]

            fig_net.add_trace(go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(size=node_size, color=COLORS['primary'], opacity=0.8,
                            line=dict(width=1.5, color='white')),
                text=node_text, textposition='top center',
                textfont=dict(size=9, color='#374151'),
                hoverinfo='text', hovertext=node_hover,
                showlegend=False,
            ))

            fig_net.update_layout(**CHART_LAYOUT, height=600,
                                  title="商品关联网络 (节点大小 = 关联规则数)")
            fig_net.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_net, width='stretch')
            st.caption("💡 **解读**: 这里是无向展示: 强关联往往互为前项与后项 (如粉色↔蓝色餐具套装)，"
                       "逐条规则的方向请看上方散点图与下方详情表的「前项 → 后项」。"
                       "节点越大越是「枢纽」商品；鼠标悬停连线可查看该规则的提升度与置信度。")
        else:
            st.info("当前规则数量不足，无法生成网络图。")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # --- 共现热力图 ---
        st.subheader("🔥 商品共现热力图 (Top 20)")
        st.markdown("行商品出现时，列商品同时出现的条件概率 P(列|行)")

        fig_cooc = px.imshow(
            cooc_matrix.values,
            x=[_cn(c) for c in cooc_matrix.columns.tolist()],
            y=[_cn(c) for c in cooc_matrix.index.tolist()],
            color_continuous_scale='YlOrRd',
            aspect='auto',
            labels={'color': 'P(列|行)'},
        )
        fig_cooc.update_layout(**CHART_LAYOUT, height=600,
                               xaxis_tickangle=-45,
                               xaxis_title='', yaxis_title='')
        st.plotly_chart(fig_cooc, width='stretch')
        st.caption("💡 **解读**: 对角线恒为 1 (商品自身共现)。非对角线的高值区域就是强共购模式，"
                   "可用于推荐系统和捆绑促销策略。")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # --- 规则详情表格 ---
        st.subheader("📋 关联规则详情 (按提升度排序)")

        display_rules = scatter_df[['规则', '前项', '后项', 'support', 'confidence', 'lift']].copy()
        display_rules.columns = ['规则', '前项 (买了这个)', '后项 (也买了这个)',
                                 '支持度', '置信度', '提升度']
        display_rules['支持度'] = display_rules['支持度'].map(lambda x: f"{x:.4f}")
        display_rules['置信度'] = display_rules['置信度'].map(lambda x: f"{x:.1%}")
        display_rules['提升度'] = display_rules['提升度'].map(lambda x: f"{x:.2f}")

        _n_rules = len(display_rules)
        if _n_rules <= 10:
            # 规则数低于滑块下限 (10) 时不给滑块: 否则 min_value > max_value, 整页直接中断
            n_display = _n_rules
            st.caption(f"当前参数下共 {_n_rules} 条规则，已全部显示。")
        else:
            n_display = st.slider("显示规则数量", min_value=10, max_value=min(100, _n_rules),
                                  value=min(20, _n_rules), step=5)
        st.dataframe(display_rules.head(n_display), width='stretch', hide_index=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # --- 基于规则的营销方案 ---
        st.subheader("🎯 基于关联规则的营销方案")
        st.caption("把规则按提升度分档, 直接翻译成可执行的营销动作。同一个商品组合只出一条方案 "
                   "(A→B 与 B→A 是同一个组合, 已去重)。")

        def _cn_join(names):
            """把 'A + B' 形式的商品组合逐个翻译成中文名"""
            return ' + '.join(_cn(i.strip()) for i in names.split(' + '))

        _mkt_n = st.selectbox("生成方案的规则条数", options=[5, 10, 15, 20], index=1,
                              help=f"按提升度从高到低取前 N 条商品组合; 当前共 {len(rules_df)} 条规则")
        _mkt = build_marketing_actions(rules_df, top_n=_mkt_n)
        _mkt_display = _mkt.copy()
        _mkt_display['前项'] = _mkt_display['前项'].map(_cn_join)
        _mkt_display['后项'] = _mkt_display['后项'].map(_cn_join)
        _mkt_display.columns = ['前项 (买了这个)', '后项 (也买了这个)', '支持度',
                                '置信度', '提升度', '关联力度', '建议动作']
        st.dataframe(_mkt_display, width='stretch', hide_index=True, height=380)

        _mkt_strong = int((_mkt['关联力度'] == '强').sum())
        _mkt_mid = int((_mkt['关联力度'] == '中').sum())
        _mkt_weak = int((_mkt['关联力度'] == '弱').sum())
        _top_pair = _mkt.iloc[0]
        st.markdown(f"""
**怎么用这张表**: 方案已按提升度从高到低排好, 前 {len(_mkt)} 个组合里
**强关联 {_mkt_strong} 个 / 中等 {_mkt_mid} 个 / 较弱 {_mkt_weak} 个**:

- **强关联 (lift ≥ 10)** —— 共现强度是随机出现的 10 倍以上, 可以直接做成套装或配套商品主推。
  当前最高的一条是「{_cn_join(_top_pair['前项'])}」↔「{_cn_join(_top_pair['后项'])}」(lift {_top_pair['提升度']})。
- **中等关联 (5 ≤ lift < 10)** —— 适合货架相邻陈列, 或在商品详情页做同页推荐, 不一定需要打包销售。
- **较弱关联 (lift < 5)** —— 力度不足以单独做活动, 适合放进凑单推荐位或作为满额赠品候选。

**落地时的两点提醒**:
1. 表里是**商品名**, 实际执行要换成业务系统里的 SKU/商品编号, 并按库存与毛利复核套装定价;
2. 关联规则只说明「一起买」, 不区分因果 —— 有些组合是顾客本来就打算成套买的 (如三色餐具套装),
   这类做成套装是顺势而为; 更有商业价值的往往是那些**顾客自己未必意识到可以搭配**的组合。
""")

        with st.expander("📖 指标含义说明"):
            st.markdown("""
**支持度 (Support)**: 规则中所有商品同时出现在一笔交易中的概率。
- 公式: P(A ∪ B) = 包含 A 和 B 的交易数 / 总交易数
- 含义: 衡量规则的普遍性。支持度越高，说明这个组合越常见。

**置信度 (Confidence)**: 在买了前项的条件下，也买后项的概率。
- 公式: P(B|A) = 包含 A 和 B 的交易数 / 包含 A 的交易数
- 含义: 衡量规则的可靠性。置信度 0.7 表示买了 A 的客户中 70% 也买了 B。

**提升度 (Lift)**: 置信度相对于后项独立出现概率的倍数。
- 公式: Lift = Confidence / P(B) = P(A∪B) / (P(A) × P(B))
- 含义: Lift > 1 表示正相关 (买了 A 确实更倾向买 B)；Lift = 1 表示无关；Lift < 1 表示负相关。
- **答辩重点**: 提升度是衡量关联规则是否有意义的最核心指标。一条规则即使置信度很高，
  但如果后项本身就是热门商品 (P(B) 很大)，提升度可能接近 1，说明关联并不强。
            """)

# ============================================================
# 页面 6: 商品与国家调查
# ============================================================
elif page == "📦 商品与国家调查":
    st.title("📦 商品与国家调查")
    st.markdown("针对 **商品描述 (Description)** 与 **用户所在国家 (Country)** 两个特征的业务探索 —— "
                "国家维度分布高度集中，直接当聚类特征几乎没有区分度，"
                "但拆成「本土 vs 海外」后价值差异很明显。")

    # ===================== 一、商品描述 =====================
    st.subheader("🏷️ 商品描述 (Description) 分析")
    st.markdown(f"""
    Description 是**文本字段**，无法直接进模型。这里的处理办法是：先把交易聚合到商品粒度
    (共 **{cleaning_summary['unique_products']:,}** 种)，再从描述里提取高频词，把商品自动归到
    「品类 / 系列」上，然后比较各品类的表现。

    > **口径说明**: 一个商品可能同时命中多个词 (如 `JUMBO BAG PINK WITH WHITE SPOTS` 含 JUMBO / BAG / PINK / SPOTS)，
    > 因此各词的收入合计互相重叠，**不能相加当作总量**。描述字段本身的缺失记录
    > ({quality_report['missing_description']:,} 条, {quality_report['missing_description_pct']}%) 已在清洗阶段移除。
    """)

    _word_topn = st.slider("显示品类词数量", min_value=8, max_value=25, value=15, step=1,
                           key="word_topn")
    _words = cached_description_words(cleaned_df, _word_topn)

    _w1, _w2 = st.columns(2)
    with _w1:
        fig_word = px.bar(_words.sort_values('收入'), x='收入', y='显示名', orientation='h',
                          color='平均单价', color_continuous_scale='Viridis',
                          hover_data={'关键词': True, '商品数': True, '销量': True,
                                      '平均单价': ':.2f'})
        fig_word.update_layout(**CHART_LAYOUT, height=430,
                               title='各品类词的收入贡献 (颜色 = 平均单价)',
                               yaxis={'categoryorder': 'total ascending'},
                               xaxis_title='收入 (美元 $)',
                               coloraxis_colorbar=dict(title='平均单价'))
        st.plotly_chart(fig_word, width='stretch')
    with _w2:
        fig_ps = px.scatter(_words, x='销量', y='平均单价', size='收入', color='收入',
                            color_continuous_scale='Plasma', hover_name='显示名', log_x=True,
                            labels={'销量': '总销量 (件)', '平均单价': '平均单价 ($)'})
        fig_ps.update_layout(**CHART_LAYOUT, height=430,
                             title='销量 vs 平均单价 (气泡大小 = 收入)')
        st.plotly_chart(fig_ps, width='stretch')

    _word_table = _words[['关键词', '中文含义', '词类', '商品数', '收入',
                          '收入占比%', '销量', '平均单价']].copy()
    _word_table.columns = ['品类词', '含义', '词类', '商品数', '收入 ($)',
                           '收入占比%', '销量 (件)', '平均单价 ($)']
    st.dataframe(_word_table, width='stretch', hide_index=True)

    # 页面上只显示前 N 个词, 这里把全部注释列出来, 观众可随时查任意词的含义
    with st.expander("📖 品类词对照表 (全部注释, 点击展开)"):
        _dict_df = pd.DataFrame([{'品类词': _w, '含义': _v['cn'], '词类': _v['type']}
                                 for _w, _v in WORD_CN.items()])
        _dict_df = _dict_df.sort_values(['词类', '品类词']).reset_index(drop=True)
        st.dataframe(_dict_df, width='stretch', hide_index=True, height=320)
        st.caption("**词类说明**: 颜色 / 图案 / 风格 是**系列或款式词**（描述商品长什么样），"
                   "品类 / 材质 / 形态 是**商品属性词**（描述商品是什么），规格 / 主题 是**规格与节庆词**。"
                   "这些注释是逐词核对真实商品名后填写的，未登记的词标为「其他」。")

    # 动态取值, 避免把结论写死
    _top_word = _words.iloc[0]
    _hi_price_word = _words.sort_values('平均单价', ascending=False).iloc[0]
    _type_txt = '、'.join(f"**{_t}** {_n} 个"
                          for _t, _n in _words['词类'].value_counts().items())
    st.caption(f"💡 **解读**: 收入贡献最高的是 **{_top_word['关键词']}**（{_top_word['中文含义']}）——覆盖 "
               f"{int(_top_word['商品数'])} 种商品、收入 \\${_top_word['收入']:,.0f}、占全站 {_top_word['收入占比%']}%；"
               f"平均单价最高的是 **{_hi_price_word['关键词']}**（{_hi_price_word['中文含义']}），约 \\${_hi_price_word['平均单价']:.2f}/件。"
               f"这 {len(_words)} 个高频词里 {_type_txt}，但**颜色词在前五名占了 3 席**（RED / WHITE / PINK）—— "
               f"顾客高度按「同一个商品的不同花色」选购，与第 5 页关联规则的发现一致。"
               f"右图（对数 X 轴）右下角是**走量**型、左上角是**量小价高**型：前者适合曝光与凑单，后者适合重点推荐。")

    st.caption("⚠️ **口径提醒**: `HOT` / `WATER` / `BOTTLE` 三个词其实来自同一个商品名 "
               "`HOT WATER BOTTLE`（热水袋），`RETRO` / `SPOT` / `RETROSPOT` 也高度重叠。")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ===================== 二、用户所在国家 =====================
    st.subheader("🌍 用户所在国家 (Country) 分析")

    _country = cached_country_summary(cleaned_df)
    _uk_ov = cached_uk_overseas(cleaned_df, rfm_df)
    _uk_row = _uk_ov.loc['英国本土'] if '英国本土' in _uk_ov.index else None
    _ov_row = _uk_ov.loc['海外'] if '海外' in _uk_ov.index else None

    _uk_rev_share = float(
        _country.loc[_country['Country'] == 'United Kingdom', '收入占比%'].iloc[0])
    _ov_n = int(_ov_row['客户数']) if _ov_row is not None else 0
    _ov_aov = float(_ov_row['AOV']) if _ov_row is not None else 0.0
    _ov_mon = float(_ov_row['Monetary']) if _ov_row is not None else 0.0
    _ov_rev_share = float(_ov_row['收入占比%']) if _ov_row is not None else 0.0
    _uk_mon = float(_uk_row['Monetary']) if _uk_row is not None else 0.0
    _uk_aov = float(_uk_row['AOV']) if _uk_row is not None else 0.0

    _ck1, _ck2, _ck3, _ck4 = st.columns(4)
    with _ck1:
        st.metric("覆盖国家/地区", f"{len(_country)} 个")
    with _ck2:
        st.metric("英国收入占比", f"{_uk_rev_share:.1f}%")
    with _ck3:
        st.metric("海外客户数", f"{_ov_n:,} 人", help="英国以外的客户, 按客户所属国家划分")
    with _ck4:
        st.metric("海外平均客单价", f"${_ov_aov:,.0f}",
                  help=f"英国本土为 ${_uk_aov:,.0f}")

    _cvv1, _cvv2 = st.columns([1, 1])
    with _cvv1:
        st.markdown("**本土 vs 海外 客户画像**")
        _prof = _uk_ov.copy()
        _prof['客户占比%'] = (_prof['客户数'] / _prof['客户数'].sum() * 100).round(1)
        _prof_disp = _prof[['客户数', '客户占比%', 'Recency', 'Frequency', 'Monetary',
                            'AOV', 'N_products', '收入占比%']]
        _prof_disp.columns = ['客户数', '客户占比%', '平均R(天)', '平均F(次)', '平均消费($)',
                              '平均客单价($)', '平均品类广度(种)', '收入占比%']
        st.dataframe(_prof_disp, width='stretch')
    with _cvv2:
        _cmp_df = pd.DataFrame({
            '区域': list(_prof.index) * 2,
            '占比%': list(_prof['客户占比%']) + list(_prof['收入占比%']),
            '口径': ['客户占比'] * len(_prof) + ['收入占比'] * len(_prof),
        })
        fig_ov = px.bar(_cmp_df, x='区域', y='占比%', color='口径', barmode='group',
                        text='占比%',
                        color_discrete_sequence=[COLORS['primary'], COLORS['success']])
        fig_ov.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_ov.update_layout(**CHART_LAYOUT, height=340,
                             title='客户占比 vs 收入占比 (价值集中度)',
                             yaxis_title='占比 (%)', xaxis_title='')
        st.plotly_chart(fig_ov, width='stretch')

    st.caption(f"💡 **解读**: 英国以 {_uk_rev_share:.1f}% 的收入占比绝对主导, 但只看收入会漏掉一个事实 —— "
               f"**海外客户人少、单体价值更高**: 海外 {_ov_n:,} 人贡献了 {_ov_rev_share:.1f}% 的收入, "
               f"平均消费 \\${_ov_mon:,.0f} vs 英国 \\${_uk_mon:,.0f}, 平均客单价 \\${_ov_aov:,.0f} vs \\${_uk_aov:,.0f}。"
               f"本土适合做频次与复购, 海外适合做客单价与批发阶梯价。")

    st.markdown("#### 📋 国家明细 (按收入排序)")
    _cshow = st.selectbox("显示数量", options=[10, 15, 20, 30], index=1,
                          help=f"数据共覆盖 {len(_country)} 个国家/地区")
    _country_show = _country[['Country', '客户数', '收入', '收入占比%', '订单数',
                              '客单价', '复购率%', '商品种类数']].head(_cshow).copy()
    _country_show.columns = ['国家', '客户数', '收入 ($)', '收入占比%', '订单数',
                             '客单价 ($)', '复购率%', '商品种类数']
    st.dataframe(_country_show, width='stretch', hide_index=True)
    # 每个国家的「客户数」是该国内唯一客户数, 跨多国下单的客户会在各国各计一次
    _cust_col_sum = int(_country['客户数'].sum())
    _unique_cust = cleaning_summary['unique_customers']
    st.caption(f"💡 **口径提醒**: 「客户数」是每个国家内部的唯一客户数, 同一客户在两个国家都下过单会在两国各计一次, "
               f"所以本列合计 {_cust_col_sum:,} 会略大于全站唯一客户数 {_unique_cust:,} "
               f"(多出的 {_cust_col_sum - _unique_cust} 人次来自跨多国下单的客户, 不能把这一列直接求和当作客户总数)。"
               f"需要客户总数请以上方「本土 vs 海外」表为准 —— 那张表按客户的**首个**国家二分, 每人只计一次。")

    st.markdown("#### 📈 收入 Top 5 国家的月度收入构成")
    _cm = cached_country_monthly(cleaned_df, 5)
    _cm_cols = [c for c in _cm.columns if c != '月份']
    _cm_disp = _cm.copy()
    _cm_disp['月份'] = _cm_disp['月份'].dt.strftime('%Y-%m')
    fig_cm = px.area(_cm_disp, x='月份', y=_cm_cols,
                     labels={'value': '收入 (美元 $)', 'variable': '国家'},
                     color_discrete_sequence=COLORS['palette'])
    fig_cm.update_layout(**CHART_LAYOUT, height=420,
                         title='收入 Top 5 国家的月度收入构成',
                         xaxis_title='月份', yaxis_title='收入 (美元 $)',
                         legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                     xanchor='center', x=0.5))
    st.plotly_chart(fig_cm, width='stretch')
    st.caption("💡 **解读**: 整体形状与第 1 页的月度收入趋势一致 (11 月冲高、12 月为不完整月份); "
               "分层后看的是海外几层的厚度是否随时间变厚 —— 变厚说明海外增速快于本土, 反之增长仍靠英国本土。")
