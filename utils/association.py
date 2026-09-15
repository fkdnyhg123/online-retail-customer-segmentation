"""
Association Rule Mining Module (FP-Growth)
Mines product co-purchase patterns from transaction data.
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
import networkx as nx


def prepare_basket_matrix(cleaned_df: pd.DataFrame, top_n_items: int = 150) -> pd.DataFrame:
    """
    Convert transaction data to a one-hot encoded basket matrix.

    Parameters:
        cleaned_df: cleaned transaction DataFrame (must have Invoice, StockCode, Description)
        top_n_items: only keep the N most frequent items to control memory/compute

    Returns:
        One-hot DataFrame (rows = invoices, columns = item labels, values = bool)
    """
    # Build StockCode -> short description mapping
    desc_map = (
        cleaned_df.dropna(subset=['Description'])
        .groupby('StockCode')['Description']
        .first()
        .to_dict()
    )

    # Get unique (Invoice, StockCode) pairs — deduplicate within each basket
    basket = cleaned_df[['Invoice', 'StockCode']].drop_duplicates()

    # Count item frequency and keep top N
    item_counts = basket['StockCode'].value_counts()
    top_items = item_counts.head(top_n_items).index.tolist()
    basket = basket[basket['StockCode'].isin(top_items)]

    # One-hot encode: rows = Invoice, columns = StockCode
    basket_matrix = pd.crosstab(basket['Invoice'], basket['StockCode']).astype(bool)

    # Rename columns to readable descriptions (truncate long names)
    rename = {}
    for code in basket_matrix.columns:
        desc = desc_map.get(code, code)
        # Truncate to 40 chars for readability
        short = desc[:40].strip() if len(desc) > 40 else desc.strip()
        rename[code] = f"{short}"

    basket_matrix = basket_matrix.rename(columns=rename)

    # Handle duplicate column names after renaming (append index)
    cols = basket_matrix.columns.tolist()
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c} ({seen[c]})")
        else:
            seen[c] = 0
            new_cols.append(c)
    basket_matrix.columns = new_cols

    return basket_matrix


def run_fpgrowth(basket_matrix: pd.DataFrame,
                 min_support: float = 0.02,
                 min_confidence: float = 0.3,
                 min_lift: float = 1.0) -> dict:
    """
    Run FP-Growth algorithm and generate association rules.

    Parameters:
        basket_matrix: one-hot encoded basket DataFrame
        min_support: minimum support threshold (0-1)
        min_confidence: minimum confidence threshold (0-1)
        min_lift: minimum lift threshold (>0, 1=no correlation)

    Returns:
        dict with keys:
            'itemsets': frequent itemsets DataFrame
            'rules': association rules DataFrame (may be empty if no rules pass threshold)
            'n_transactions': number of transactions analyzed
            'n_items': number of unique items considered
    """
    n_transactions = len(basket_matrix)
    n_items = basket_matrix.shape[1]

    # Find frequent itemsets
    itemsets = fpgrowth(basket_matrix, min_support=min_support, use_colnames=True)

    if itemsets.empty:
        return {
            'itemsets': itemsets,
            'rules': pd.DataFrame(),
            'n_transactions': n_transactions,
            'n_items': n_items,
        }

    # Generate association rules
    try:
        rules = association_rules(itemsets, metric='confidence',
                                  min_threshold=min_confidence,
                                  num_itemsets=len(itemsets))
    except ValueError:
        # No rules can be generated
        rules = pd.DataFrame()

    if not rules.empty:
        # Filter by lift
        rules = rules[rules['lift'] >= min_lift].copy()
        # Sort by lift descending
        rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

    return {
        'itemsets': itemsets,
        'rules': rules,
        'n_transactions': n_transactions,
        'n_items': n_items,
    }


def compute_cooccurrence_matrix(basket_matrix: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Compute item co-occurrence matrix for the top N most frequent items.

    Parameters:
        basket_matrix: one-hot encoded basket DataFrame
        top_n: number of top items to include

    Returns:
        Square DataFrame (top_n × top_n) with co-occurrence counts,
        normalized to show conditional probability P(col | row).
    """
    # Get top N items by frequency
    freq = basket_matrix.sum().sort_values(ascending=False)
    top_items = freq.head(top_n).index.tolist()

    # Subset to top items
    sub = basket_matrix[top_items].astype(int)

    # Co-occurrence = matrix multiplication
    cooc = sub.T @ sub

    # Normalize: P(j|i) = cooc[i,j] / cooc[i,i]
    diag = np.diag(cooc.values).copy()
    diag[diag == 0] = 1  # avoid division by zero
    cooc_norm = cooc.values / diag[:, np.newaxis]

    result = pd.DataFrame(cooc_norm, index=top_items, columns=top_items)
    return result


def build_network_graph(rules_df: pd.DataFrame, top_n: int = 30) -> dict:
    """
    Build a network graph from association rules.

    Parameters:
        rules_df: association rules DataFrame
        top_n: maximum number of rules to include

    Returns:
        dict with 'nodes' (list of dicts: id, label, degree, support)
        and 'edges' (list of dicts: source, target, lift, confidence, support)
        and 'pos' (dict: node_id -> (x, y) positions from spring layout)
    """
    if rules_df.empty:
        return {'nodes': [], 'edges': [], 'pos': {}}

    # Take top N rules by lift
    top_rules = rules_df.head(top_n)

    G = nx.DiGraph()

    for _, row in top_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])

        for a in antecedents:
            if not G.has_node(a):
                G.add_node(a, support=0)
        for c in consequents:
            if not G.has_node(c):
                G.add_node(c, support=0)

        # Edge from each antecedent to each consequent
        for a in antecedents:
            for c in consequents:
                if G.has_edge(a, c):
                    # Keep the stronger relationship
                    existing = G[a][c]
                    if row['lift'] > existing['lift']:
                        G[a][c] = {
                            'lift': row['lift'],
                            'confidence': row['confidence'],
                            'support': row['support'],
                        }
                else:
                    G.add_edge(a, c,
                               lift=row['lift'],
                               confidence=row['confidence'],
                               support=row['support'])

    # Update node support (max support of connected edges)
    for node in G.nodes():
        max_sup = 0
        for u, v, data in G.edges(node, data=True):
            max_sup = max(max_sup, data.get('support', 0))
        G.nodes[node]['support'] = max_sup
        G.nodes[node]['degree'] = G.degree(node)

    # Compute spring layout
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    nodes = [{'id': n, 'label': n, **G.nodes[n]} for n in G.nodes()]
    edges = [{'source': u, 'target': v, **d} for u, v, d in G.edges(data=True)]

    return {'nodes': nodes, 'edges': edges, 'pos': pos}


def threshold_sweep(basket_matrix: pd.DataFrame,
                    supports: list, confidences: list,
                    min_lift: float = 1.0) -> tuple:
    """
    扫描「支持度 × 置信度」网格, 为"阈值范围该定在哪"提供数据依据。

    优化依据: 频繁项集对 support 单调 —— 低阈值下的项集集合必然包含高阈值下的,
    因此只需在最低支持度上跑一次 fpgrowth, 再按 support 过滤 + 逐置信度生成规则,
    避免在几十个阈值组合上重复挖掘。

    Parameters:
        basket_matrix: 购物篮矩阵
        supports: 待扫描的支持度列表 (建议含滑块的两端)
        confidences: 待扫描的置信度列表
        min_lift: 提升度下限 (默认 1.0, 即只保留正相关规则)

    Returns:
        (DataFrame, dict)
        - DataFrame: 每行一个组合 —— 支持度, 置信度, 频繁项集数, 规则数, 最高提升度, 最高置信度
        - dict: 支持度上界的两个依据
            'max_item_support': 最高频单品自身的支持度
            'max_pair_support': 最高频**商品对**的支持度 —— 规则的天然上界, 因为一条规则
                                至少需要两种商品共现; 阈值超过它必然一条规则都挖不出来
          (两者都由 min(supports) 这一次挖掘结果统计得出, 故 supports 的最小值需足够低)
    """
    cols = ['支持度', '置信度', '频繁项集数', '规则数', '最高提升度', '最高置信度']
    info = {'max_item_support': 0.0, 'max_pair_support': 0.0}
    if basket_matrix.empty:
        return pd.DataFrame(columns=cols), info

    base = fpgrowth(basket_matrix, min_support=min(supports), use_colnames=True)

    info['max_item_support'] = float(basket_matrix.mean().max())
    if not base.empty:
        _n_items = base['itemsets'].apply(len)
        _pairs = base[_n_items >= 2]
        if not _pairs.empty:
            info['max_pair_support'] = float(_pairs['support'].max())

    rows = []
    for s in supports:
        sub = base[base['support'] >= s]
        for c in confidences:
            n_rules, max_lift, max_conf = 0, np.nan, np.nan
            if not sub.empty:
                try:
                    r = association_rules(sub, metric='confidence',
                                          min_threshold=c, num_itemsets=len(sub))
                except ValueError:
                    r = pd.DataFrame()
                if not r.empty:
                    r = r[r['lift'] >= min_lift]
                if not r.empty:
                    n_rules = int(len(r))
                    max_lift = round(float(r['lift'].max()), 2)
                    max_conf = round(float(r['confidence'].max()), 3)
            rows.append({
                '支持度': s, '置信度': c, '频繁项集数': int(len(sub)),
                '规则数': n_rules, '最高提升度': max_lift, '最高置信度': max_conf,
            })
    return pd.DataFrame(rows, columns=cols), info


def build_marketing_actions(rules_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    把关联规则翻译成可执行的营销动作, 按提升度分档决定力度与形式。

    分档依据 lift 而非 confidence: 置信度会被后项自身的高频次抬高 (热门商品天然
    容易"被一起买"), lift 才是"比随机共现强多少倍"的强度指标。
      lift >= 10  -> 强: 直接组合成套装/配套商品主推
      5 <= lift < 10 -> 中: 货架相邻陈列 / 详情页同页推荐
      lift < 5  -> 弱: 凑单推荐位 / 满额赠品候选

    Parameters:
        rules_df: 规则表 (已按 lift 降序)
        top_n: 取前 N 条规则生成方案

    Returns:
        DataFrame: 前项, 后项, 支持度, 置信度, 提升度, 关联力度, 建议动作
    """
    if rules_df.empty:
        return pd.DataFrame()

    rows = []
    seen_pairs = set()
    for _, r in rules_df.iterrows():
        # A→B 与 B→A 是同一个商品组合 (实测两向 lift 相同), 做方案时只保留一条, 否则
        # 同一个套装会被列出两次, 让方案表虚增一倍
        _pair = frozenset(r['antecedents']) | frozenset(r['consequents'])
        if _pair in seen_pairs:
            continue
        seen_pairs.add(_pair)

        if r['lift'] >= 10:
            strength, action = '强', '组合成套装/配套商品主推'
        elif r['lift'] >= 5:
            strength, action = '中', '货架相邻陈列 / 详情页同页推荐'
        else:
            strength, action = '弱', '凑单推荐位 / 满额赠品候选'
        rows.append({
            '前项': ' + '.join(sorted(r['antecedents'])),
            '后项': ' + '.join(sorted(r['consequents'])),
            '支持度': round(float(r['support']), 4),
            '置信度': round(float(r['confidence']), 3),
            '提升度': round(float(r['lift']), 2),
            '关联力度': strength,
            '建议动作': action,
        })
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)
