"""
Category-level Association Rules Module (报告 6.2 节配套)

把 4,011 个 SKU 归到互斥的单一品类, 再按发票聚合成「品类购物篮」跑 FP-Growth。
与 utils/association.py 的单品级挖掘共用同一套指标定义, 便于 6.2.3 做粒度对比。

设计约束 (为什么必须互斥单品类):
  - 一个商品可以命中多个描述词 (HOT WATER BOTTLE 同时含 HOT/WATER/BOTTLE),
    若允许多标签, 同一商品会把多个品类列同时置 1, 造出「自己和自己共现」的伪规则;
  - 因此这里用「按顺序首个命中即归类」的优先级规则表, 每个 SKU 只有一个品类。

关键词表按 Description 高频词人工整理, 收入覆盖率实测 ~93% (未命中归入「其他」,
挖掘时剔除「其他」, 避免它作为万能类污染所有规则)。
"""
import numpy as np
import pandas as pd

# (品类, [关键词...]) —— 自上而下, 首个命中即归类, 互斥。
# 顺序与词形都有讲究, 改之前先看这几条已踩过的坑:
#   1) 词要用词干: BAKE 匹配不到 BAKING (动词变形掉 e), 得写 BAK;
#   2) 子串陷阱: 'TIN' 会命中 BUNTING、'CHILD' 会把 CHILDS UMBRELLA 塞进玩具、
#      'FAIRY' 会把 FAIRY CAKE 蛋糕纸杯塞进烛台 —— 所以 TIN 带空格、CHILD/KIDS 删掉、
#      餐厨排在烛台前面、家居(伞/垫/毯)排在玩具前面;
#   3) 家居保暖不放 BATH/WALL/BRUSH 这类宽词, 否则会截走玩具(BATH SPLASH)与文具(PAINT BRUSH)。
CATEGORY_RULES = [
    ('首饰收纳盒', ['NECKLACE', 'BRACELET', 'EARRING', 'JEWELLERY', 'TRINKET',
                     'CHARM', 'PENDANT', 'COMPACT']),
    ('相框相册',   ['FRAME', 'PHOTO', 'PICTURE', 'ALBUM', 'RECORD COVER', 'CANVAS PRINT']),
    ('包袋',       ['BAG', 'POUCH', 'PURSE', 'SHOPPING', 'SHOPPER', 'CARRY ALL',
                     'NECESSAIRE', 'VANITY']),
    ('家居保暖',   ['CUSHION', 'CUSHARE', 'BLANKET', 'FLANNEL', 'RUG', 'DOORMAT', 'DOOR MAT',
                     'PARASOL', 'UMBRELLA', 'CURTAIN', 'PILLOW', 'CLOCK', 'COAT',
                     'SHOE', 'SLIPPER', 'SOCK', 'SOMBRERO', 'SCARF', 'TOWEL',
                     'HOT WATER', 'HAND WARMER', 'WOOLLY', 'HOTTIE', 'MIRROR', 'CUSH', 'MATCH']),
    ('餐厨烘焙',   ['CAKE', 'BAK', 'PLATE', 'CUP', 'MUG', 'BOWL', 'NAPKIN', 'CUTLER',
                     'COFFEE', 'TRAY', 'JUG', 'COASTER', 'PLACEMAT', 'TABLECLOTH',
                     'DISH', 'KITCHEN', 'CUTTING', 'FORK', 'SPOON', 'ESPRESSO',
                     'TEAPOT', 'SUGAR', 'PORRINGER', 'CHOCOLATE', 'WINE', ' GIN ', 'GIN +',
                     'CHEESE', 'BREAD', 'SALT', 'PEPPER', 'DOIL', 'SERVING',
                     'TEA GLASS', 'TABLE CLOTH',
                     'DRINK', 'APRON', 'TISSUE', 'BREAKFAST SET', 'COOKIE CUTTERS',
                     'COOKING SET', 'RECIPE', 'TEATIME', 'TEA SET', 'TEACUP',
                     'FRYING', 'TUMBLER', 'PLATTER']),
    ('烛台灯具',   ['T-LIGHT', 'TEALIGHT', 'CANDL', 'LANTERN', 'FAIRY', 'LAMP',
                     'LIGHT', 'HURRICANE']),
    ('玩具游戏',   ['TOY', 'GAME', 'PUZZLE', 'JIGSAW', 'DOLL', 'KITE', 'SKATEBOARD',
                     'ROBO', 'CARAVAN', 'PLAYING', 'DOMINO', 'BUILDING BLOCK',
                     'GLIDER', 'PRINCESS', 'SOLDIER', 'SKITTLES', 'SPACEBOY',
                     'CIRCUS', 'WINDMILL', 'GLOBE', 'WHEELBARROW']),
    ('文具手工',   ['WRAP', 'STATIONERY', 'PENCIL', 'NOTEBOOK', 'JOURNAL', 'MEMO',
                     'STAMP', 'STICKER', 'CHALK', 'GREETING', 'BILLFOLD', 'MESSAGE',
                     'PAPERCLIP', 'DIARY', 'BLACK BOARD', 'PAINT', 'CLAY', 'FELTCRAFT',
                     'SEWING', 'ENVELOPE', 'SCISSOR', 'PENS', 'STENCIL', 'LABEL',
                     ' PEN', 'PEN.', 'PEN ', 'ERASER', 'RULER']),
    ('收纳挂置',   ['BOX', 'HOLDER', 'HOLDR', 'HLDR', 'BIN', ' TIN', 'TIN ', 'JAR',
                     'CADDY', 'HOOK', 'PEG', 'RACK', 'STAND', 'ORGANISER', 'STORAGE',
                     'CHEST', 'CASE', 'CONTAINER', 'COSMETIC', 'UTENSIL', 'KEY RACK',
                     'DRAWER KNOB', 'PIGGY']),
    ('派对节庆',   ['PARTY', 'BUNTING', 'CHRISTMAS', 'EASTER', 'VALENTINE', 'BIRTHDAY',
                     'HALLOWEEN', 'WREATH', 'GARLAND', 'BELL', 'CHAIN KIT']),
    ('装饰园艺',   ['ORNAMENT', 'FIGURINE', 'STATUE', 'GARDEN', 'BIRDBATH', 'DECOR',
                     'FLAG', 'WINDCHIME', 'BIRD', 'ANIMAL', 'DOG', 'CAT', 'RABBIT',
                     'HEN', 'FOX', 'OWL', 'LAMB', 'BUTTERFLY', 'FLOWER', 'PLANT',
                     'VASE', 'SOAP', 'SIGN', 'TRELLIS', 'WICKER', 'FILIGREE', 'DOVE',
                     'DUCK', 'MICE', 'MOUSE', 'STAR', 'SUN', 'MOON', 'HEART', 'LOVE',
                     'PLEASE', 'METAL', 'WOOD', 'VINTAGE', 'FAN', 'ROPE', 'GARDE']),
]

OTHER = '其他'


def assign_category(description: str) -> str:
    """按 CATEGORY_RULES 顺序给单个商品描述归类, 未命中返回「其他」。"""
    d = str(description).upper()
    for cat, kws in CATEGORY_RULES:
        for k in kws:
            if k in d:
                return cat
    return OTHER


def sku_category_table(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """SKU → 品类映射表 (含收入, 供人工核对与覆盖率报告)。"""
    t = (cleaned_df.groupby('StockCode')
         .agg(Description=('Description', 'first'), Revenue=('Revenue', 'sum'),
              发票数=('Invoice', 'nunique'))
         .reset_index())
    t['Category'] = t['Description'].map(assign_category)
    return t


def category_coverage(cleaned_df: pd.DataFrame) -> dict:
    """映射覆盖率与各类收入占比。"""
    t = sku_category_table(cleaned_df)
    total = float(t['Revenue'].sum())
    mapped = t.loc[t['Category'] != OTHER, 'Revenue'].sum()
    by_cat = (t.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
              / total * 100).round(2)
    return {
        'n_sku': int(len(t)),
        'n_category': int(t.loc[t['Category'] != OTHER, 'Category'].nunique()),
        'coverage_pct': round(mapped / total * 100, 2),
        'other_pct': round(100 - mapped / total * 100, 2),
        'other_sku': int((t['Category'] == OTHER).sum()),
        'revenue_pct_by_category': by_cat.to_dict(),
    }


def build_category_basket(cleaned_df: pd.DataFrame, drop_other: bool = True) -> pd.DataFrame:
    """发票 × 品类 的 0/1 购物篮矩阵 (同一发票内同一品类只算一次, 不看数量)。"""
    df = cleaned_df.copy()
    df['Category'] = df['Description'].map(assign_category)
    if drop_other:
        df = df[df['Category'] != OTHER]
    ct = pd.crosstab(df['Invoice'], df['Category'])
    return (ct > 0)


def mine_category_rules(basket: pd.DataFrame, min_support: float = 0.05,
                        min_confidence: float = 0.30, max_len: int = 2) -> dict:
    """品类级 FP-Growth, 与单品级共用 lift 口径。"""
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    fi = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    if len(fi) < 2:
        return {'itemsets': fi, 'rules': pd.DataFrame(), 'n_transactions': len(basket)}
    rr = association_rules(fi, metric='lift', min_threshold=1.0)
    rr = rr[rr['confidence'] >= min_confidence].sort_values('lift', ascending=False).reset_index(drop=True)
    return {'itemsets': fi, 'rules': rr, 'n_transactions': len(basket)}


def category_pair_table(basket: pd.DataFrame, permutations: int = 200,
                        random_state: int = 42) -> pd.DataFrame:
    """全部品类两两组合的关联指标 + **篮子大小校正**。

    为什么必须校正: 一张发票平均含 5 个品类 (共 11 类), 任意两类的共现概率天然高于
    独立假设, 于是原始提升度会全线 >1 —— 那是篮子大小的算术结果, 不是搭配信号。

    零假设: 保留每张发票的品类数 k, 按品类边际分布**无放回**随机重选 k 类
    (Gumbel top-k 加权抽样), 置换 permutations 次取期望共现支持度。
      校正后提升度 = 原始提升度 ÷ (期望支持度 / 边际乘积)   —— 跌破 1 即被证伪
      校正z值      = (观察支持度 - 期望支持度) / 期望支持度的标准误

    Returns:
        DataFrame 按校正后提升度降序: 品类A, 品类B, 共现发票数, 支持度, 提升度,
        P(B|A), P(A|B), 期望支持度, 校正后提升度, 校正z值
    """
    import itertools
    cats = list(basket.columns)
    K, n = len(cats), len(basket)
    X = basket.values.astype(np.int64)
    co = X.T @ X
    marg = X.sum(0) / n
    ks = X.sum(1)

    # ---- 置换零模型 ----
    rng = np.random.default_rng(random_state)
    logp = np.log(np.maximum(marg / marg.sum(), 1e-12))
    exp_co = np.zeros((K, K))
    for k in np.unique(ks):
        idx = np.where(ks == k)[0]
        kk = int(k)
        if kk < 2:
            continue
        if kk >= K:                                          # 该发票已含全部品类, 无需抽样
            exp_co += np.ones((K, K)) * len(idx)
            continue
        cnt_k = np.zeros((K, K))
        for _ in range(permutations):
            # 每轮重新抽 Gumbel 噪声; 矩阵必须用 int64 —— int8/int16 在累加时会溢出,
            # 把期望支持度算成垃圾值 (校正后提升度会虚高到几百倍)
            G = -np.log(-np.log(rng.random((len(idx), K)))) + logp
            pick = np.argpartition(-G, kk, axis=1)[:, :kk]
            b = np.zeros((len(idx), K), dtype=np.int64)
            b[np.repeat(np.arange(len(idx)), kk), pick.reshape(-1)] = 1
            cnt_k += b.T @ b
        exp_co += cnt_k / permutations

    rows = []
    for i, j in itertools.combinations(range(K), 2):
        s = co[i][j] / n
        es = exp_co[i][j] / n
        base = marg[i] * marg[j]
        se = max(np.sqrt(es * (1 - es) / n), 1e-9)
        rows.append({'品类A': cats[i], '品类B': cats[j], '共现发票数': int(co[i][j]),
                     '支持度': s, '提升度': s / base,
                     'P(B|A)': s / marg[i], 'P(A|B)': s / marg[j],
                     '期望支持度': es, '校正后提升度': (s / base) / (es / base) if es > 0 else np.nan,
                     '校正z值': (s - es) / se})
    return pd.DataFrame(rows).sort_values('校正后提升度', ascending=False).reset_index(drop=True)


def within_category_share(cleaned_df: pd.DataFrame, item_rules: pd.DataFrame,
                          top_n: int = 20) -> tuple:
    """单品级 Top-N 规则中，前项与后项**属于同一品类**的条数 —— 6.2.3 粒度对比的核心数字。
    这些规则在品类粒度下会变成「类内搭配」，跨品类挖掘结构上看不到。

    Returns: (同品类条数, 实际统计的规则条数)
    """
    desc_to_sku = {}
    for s, d in cleaned_df[['StockCode', 'Description']].drop_duplicates().values:
        if isinstance(d, str):
            desc_to_sku.setdefault(d.strip().upper(), s)
    cat_of = dict(zip(sku_category_table(cleaned_df)['StockCode'],
                      sku_category_table(cleaned_df)['Category']))
    r = item_rules.sort_values('lift', ascending=False).head(top_n)
    same = 0
    for ante, cons in zip(r['antecedents'], r['consequents']):
        ca = cat_of.get(desc_to_sku.get(str(list(ante)[0]).strip().upper()))
        cb = cat_of.get(desc_to_sku.get(str(list(cons)[0]).strip().upper()))
        if ca is not None and ca == cb:
            same += 1
    return same, len(r)


if __name__ == '__main__':
    import os
    from utils.data_cleaner import load_cleaned
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(category_coverage(load_cleaned(_repo)))
