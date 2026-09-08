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
