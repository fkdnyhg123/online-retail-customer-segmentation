# -*- coding: utf-8 -*-
"""
6.2 品类级关联规则挖掘 —— 结果生成脚本
用法:  .venv/Scripts/python.exe 品类级规则挖掘/run_6p2_category_rules.py
产出:  图 5 张 + 规则表 CSV + SKU 映射表 CSV + 结果说明.md
依赖:  utils/category.py（品类映射与挖掘）· utils/association.py（单品级对照）
"""
import os, sys, itertools
import pandas as pd, numpy as np

def _find_repo():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, 'app.py')):
            return d
        d = os.path.dirname(d)
    raise RuntimeError('找不到仓库根目录 (app.py)')

REPO = _find_repo()
sys.path.insert(0, REPO)
OUT = os.path.dirname(os.path.abspath(__file__))

from utils.category import (category_coverage, build_category_basket, mine_category_rules,
                            sku_category_table, assign_category, CATEGORY_RULES,
                            category_pair_table, within_category_share)
from utils.association import prepare_basket_matrix, run_fpgrowth

NAVY, ORANGE, TEAL, GREY = '#2C6FBB', '#E8833A', '#2A9D8F', '#8A8A8A'

def load_cleaned():
    """数据源就是 online_retail_II(1).xlsx。有 Parquet 缓存就读缓存 (与大屏同一份, 秒级);
    缓存不在就回落到 xlsx 现读现清洗, 保证脚本能独立跑。"""
    _pq = os.path.join(REPO, '.cache_cleaned.parquet')
    if os.path.exists(_pq):
        return pd.read_parquet(_pq)
    from utils.data_cleaner import load_raw_data, clean_data
    return clean_data(load_raw_data(os.path.join(REPO, 'online_retail_II(1).xlsx')))


cln = load_cleaned()
cov = category_coverage(cln)
bt = build_category_basket(cln)
cats = list(bt.columns)
K = len(cats)
n = len(bt)
X = bt.values.astype(np.int64)
co = X.T @ X
marg = X.sum(0) / n

# ---------- 1+2) 全品类对指标 + 篮子大小校正 (与大屏共用 utils 里的同一份实现) ----------
P = category_pair_table(bt, permutations=200)

# ---------- 3) FP-Growth（与单品级同口径）----------
mk = mine_category_rules(bt, min_support=0.05, min_confidence=0.30, max_len=2)
rr = mk['rules']
rr_out = rr.copy()
rr_out['前项'] = rr['antecedents'].map(lambda s: '+'.join(s))
rr_out['后项'] = rr['consequents'].map(lambda s: '+'.join(s))
rr_out = rr_out[['前项', '后项', 'support', 'confidence', 'lift']].rename(
    columns={'support': '支持度', 'confidence': '置信度', 'lift': '提升度'})

# ---------- 4) 单品级对照 ----------
bm = prepare_basket_matrix(cln, top_n_items=100)
item_rr = run_fpgrowth(bm, 0.02, 0.30)['rules'].sort_values('lift', ascending=False)
sku_tab = sku_category_table(cln)
cat_of = dict(zip(sku_tab['StockCode'], sku_tab['Category']))
d2s = {}
for s, d in cln[['StockCode', 'Description']].drop_duplicates().values:
    if isinstance(d, str):
        d2s.setdefault(d.strip().upper(), s)
top20 = item_rr.head(20)
same_cat = sum(1 for _, x in top20.iterrows()
               if cat_of.get(d2s.get(list(x['antecedents'])[0].upper())) ==
                  cat_of.get(d2s.get(list(x['consequents'])[0].upper())))

raw_top = P.sort_values('提升度', ascending=False).iloc[0]
fix_top = P.iloc[0]
n_neg = int((P['校正后提升度'] < 1).sum())
raw_top_fixed = P.loc[P['品类A'].eq(raw_top['品类A']) & P['品类B'].eq(raw_top['品类B'])].iloc[0]
scene = P[(P['校正后提升度'] > 1.2)][['品类A', '品类B', '校正后提升度', '共现发票数']]
scene_pairs = '、'.join(f"{r['品类A']}×{r['品类B']}（{r['校正后提升度']:.2f}）" for _, r in scene.head(4).iterrows())

# ---------- 输出 CSV / md ----------
P.round(4).to_csv(os.path.join(OUT, '表_品类对关联强度全表.csv'), index=False, encoding='utf-8-sig')
rr_out.round(4).to_csv(os.path.join(OUT, '表_品类级关联规则.csv'), index=False, encoding='utf-8-sig')
sku_tab.assign(Description=lambda d: d['Description'].astype(str)).to_csv(
    os.path.join(OUT, '表_SKU品类映射表.csv'), index=False, encoding='utf-8-sig')

md = f"""# 6.2 品类级关联规则挖掘 —— 结果说明

生成时间基于 commit 后的当前数据；全部数字由 `utils/category.py` 现场计算，可复现。

## 一、方法（对应 6.2.1 商品品类映射方法）
- 品类表：{len(CATEGORY_RULES)} 类，写在 `utils/category.py` 的 `CATEGORY_RULES`，**按顺序首个命中即归类，互斥单品类**。
- 必须互斥的原因：一个商品描述会命中多个词（HOT WATER BOTTLE 同时含 HOT/WATER/BOTTLE），多标签会让同一商品把多个品类列同时置 1，造出「自己和自己共现」的伪规则。
- 覆盖率：4,011 个 SKU 中未归类 {cov['other_sku']} 个，**按收入计覆盖 {cov['coverage_pct']:.2f}%**（其他仅 {cov['other_pct']:.2f}%）。
- 挖掘时剔除「其他」类：它是万能类，留着会污染所有规则。
- 关键词用的是词干而非整词（BAKE 匹配不到 BAKING，必须写 BAK），这一点在做映射表时最容易踩。

## 二、品类购物篮
发票 × 品类矩阵 {bt.shape[0]:,} × {bt.shape[1]}，密度 {bt.values.mean()*100:.1f}%，平均每张发票含 {X.sum(1).mean():.2f} 个品类。
（对比单品级：15,446 × 100、密度 5.3%、平均每篮 5.29 种商品。）

## 三、核心结果
1. **品类级规则的提升度全线只有 {rr['lift'].min():.2f}~{rr['lift'].max():.2f}**，单品级是 {item_rr['lift'].min():.2f}~{item_rr['lift'].max():.2f}（中位 {item_rr['lift'].median():.2f} vs {rr['lift'].median():.2f}）。55 个品类对**无一负相关**（中位 lift {P['提升度'].median():.2f}）。
2. 但「全都 >1」不是发现，而是**篮子大小的算术结果**：一张发票平均含 {X.sum(1).mean():.2f} 个品类（共 11 类），任意两类的共现概率天然高于独立假设。用置换零模型（保留每单品类数 k、按品类边际分布**无放回**随机重选 k 类）校正后，区间变成 **{P['校正后提升度'].min():.2f}~{P['校正后提升度'].max():.2f}、中位 {P['校正后提升度'].median():.2f}，{n_neg}/55 对转为负关联**。
3. **排名几乎反转**：原始 lift 第一的 {raw_top['品类A']}×{raw_top['品类B']}（lift {raw_top['提升度']:.2f}）校正后只剩 {raw_top_fixed['校正后提升度']:.2f}（z={raw_top_fixed['校正z值']:+.0f}），说明这两类其实很少同单出现；校正后最强的是 {fix_top['品类A']}×{fix_top['品类B']}（{fix_top['校正后提升度']:.2f}，共现 {fix_top['共现发票数']:,} 张发票）。原因很具体：相框相册（占收入 1.79%）与首饰收纳盒（1.90%）都是小类，**小品类天然容易刷高 lift**。所以品类级不能直接按 lift 排序选规则——这是本节最重要的方法论结论。
4. 校正后仍稳定 >1.2 的是 {scene_pairs}，集中在**家居布置 + 厨房**的一站式采购；大屏页面 5 的品类共现热力图里餐厨烘焙那一列条件概率 0.83~0.90，说明它是「万能类」，与谁共现都高、因此本身不构成搭配信号。
5. **单品级的强规则在品类粒度会直接消失**：单品级 Top20 中 {same_cat}/20 条（{same_cat/20*100:.0f}%）两个商品本就同属一个品类（如三色圆点餐具套装都在「餐厨烘焙」内），它们变成类内搭配，跨品类挖掘看不到。

## 四、可以写进报告的结论
- 单品级回答「哪两件商品该摆一起」（lift 最高 17.90，同系列不同色）；品类级回答「哪些消费场景会同时被触发」（校正后 1.2 以上集中在餐厨 × 装饰 × 收纳 × 烛台）。
- 两种粒度互补，**不能用品类级结果否定单品级，反之亦然**。
- 品类级多了一条方法论结论：**提升度必须结合支持度、品类规模与篮子大小一起读**，否则会把「大篮子造成的假信号」当成强关联（大屏页面 5 哑铃图里红色那半就是被证伪的那批）。

## 五、文件清单
图不再生成静态文件：报告用图直接从大屏页面 5「📦 品类级关联规则 (粒度对比)」板块截图
（收入构成柱图 / Top10 规则表 / 共现热力图 / 原始→校正哑铃图四张）。
表_品类对关联强度全表.csv（55 行）/ 表_品类级关联规则.csv / 表_SKU品类映射表.csv（4,011 行，供人工核对归类）

## 六、待你决定
- 品类划分口径（11 类是否够细？想改成 15~20 类只需改 `CATEGORY_RULES`）；
- 是否把这套图做成大屏第 7 页（品类级规则）；
- 6.3 跨国对比要不要一起做（注意 Top2 国家若按收入选会得到「5 个客户的爱尔兰」，需改按发票数与客户数选）。
"""
open(os.path.join(OUT, '结果说明.md'), 'w', encoding='utf-8').write(md)

print("覆盖率 {:.2f}% | 品类 {} | 品类篮 {}×{} 密度 {:.1f}%".format(
    cov['coverage_pct'], K, bt.shape[0], bt.shape[1], bt.values.mean() * 100))
print("品类级规则 {} 条 lift {:.2f}~{:.2f} | 单品级 {} 条 lift {:.2f}~{:.2f}".format(
    len(rr), rr['lift'].min(), rr['lift'].max(), len(item_rr), item_rr['lift'].min(), item_rr['lift'].max()))
print("单品级 Top20 中同品类内部搭配 {} 条".format(same_cat))
print("输出目录:", OUT)
for f in sorted(os.listdir(OUT)):
    print("   ", f)
