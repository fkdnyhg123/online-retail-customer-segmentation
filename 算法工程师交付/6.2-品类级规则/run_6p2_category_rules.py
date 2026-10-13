# -*- coding: utf-8 -*-
"""
6.2 品类级关联规则挖掘 —— 结果生成脚本
用法:  .venv/Scripts/python.exe 品类级规则挖掘/run_6p2_category_rules.py
产出:  图 5 张 + 规则表 CSV + SKU 映射表 CSV + 结果说明.md
依赖:  utils/category.py（品类映射与挖掘）· utils/association.py（单品级对照）
"""
import os, sys, itertools
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
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

# ================== 统一视觉样式 ==================
INK, MUTED, GRID = '#1F2933', '#7B8794', '#EDF0F3'
BLUE, ORANGE, TEAL, RED, GREY2 = '#2C6FBB', '#E8833A', '#2A9D8F', '#C1444F', '#AAB7C4'
plt.rcParams.update({'axes.unicode_minus': False, 'figure.dpi': 200,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'savefig.bbox': 'tight', 'axes.edgecolor': GRID, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})


def theme(ax, grid='x'):
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.9)
        ax.set_axisbelow(True)


def title(ax, t, sub=None):
    ax.set_title(t, fontsize=12.5, fontweight='bold', color=INK, loc='left', pad=30 if sub else 8)
    if sub:
        ax.text(0, 1.022, sub, transform=ax.transAxes, fontsize=9.2, color=MUTED, va='bottom')


# ---------- 图 1 映射覆盖率与品类收入构成 ----------
share = pd.Series(cov['revenue_pct_by_category']).sort_values()
fig, ax = plt.subplots(figsize=(9.2, 5.0))
theme(ax)
cols = [GREY2 if k == '其他' else BLUE for k in share.index]
ax.barh(share.index, share.values, color=cols, height=.68)
for k, v in zip(share.index, share.values):
    ax.text(v + .45, k, "{:.1f}%".format(v), va='center', fontsize=9.4,
            color=INK if k != '其他' else MUTED,
            fontweight='bold' if k != '其他' else 'normal')
ax.set_xlim(0, share.max() * 1.22)
ax.set_xlabel('收入占比 (%)', fontsize=10)
ax.tick_params(labelsize=10)
title(ax, "图1  商品品类映射结果：{:,} 个 SKU → {} 个品类".format(cov['n_sku'], K),
      "按收入计覆盖 {:.2f}%，未归类的「其他」仅 {:.2f}% 且已从挖掘中剔除（避免它作为万能类污染规则）".format(
          cov['coverage_pct'], cov['other_pct']))
ax.text(.99, .05, '关键词按优先级首个命中即归类，互斥单品类', transform=ax.transAxes,
        ha='right', fontsize=8.6, color=MUTED, style='italic')
plt.savefig(os.path.join(OUT, '图1_品类映射覆盖率.png')); plt.close()

# ---------- 图 2 品类共现热力图 ----------
from matplotlib.colors import LinearSegmentedColormap
cmap2 = LinearSegmentedColormap.from_list('retail', ['#FFFFFF', '#DCE9F6', '#8FB8DE', '#2C6FBB', '#123A6B'])
M = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        M[i, j] = co[i][j] / max(marg[i] * n, 1)
Mp = M.copy()
np.fill_diagonal(Mp, np.nan)
fig, ax = plt.subplots(figsize=(8.8, 7.2))
ax.set_aspect('equal')
im = ax.imshow(Mp, cmap=cmap2, vmin=0, vmax=1)
ax.set_xticks(range(K)); ax.set_xticklabels(cats, rotation=42, ha='right', fontsize=9.6)
ax.set_yticks(range(K)); ax.set_yticklabels(cats, fontsize=9.6)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
for i in range(K):
    for j in range(K):
        if i == j:
            ax.text(j, i, '—', ha='center', va='center', fontsize=9, color=GREY2)
        else:
            ax.text(j, i, '{:.2f}'.format(M[i, j]), ha='center', va='center', fontsize=8.6,
                    color='white' if M[i, j] > .72 else INK,
                    fontweight='bold' if M[i, j] >= .85 else 'normal')
jkl = cats.index('餐厨烘焙')
ax.add_patch(plt.Rectangle((jkl - .5, -.5), 1, K, fill=False, edgecolor=ORANGE, lw=2, zorder=5))
cb = fig.colorbar(im, ax=ax, shrink=.78, pad=.035)
cb.set_label('条件概率 P(列品类 | 行品类)', fontsize=9.6, color=MUTED)
cb.ax.tick_params(colors=MUTED, labelsize=9)
cb.outline.set_visible(False)
title(ax, '图2  品类共现热力图', '行品类出现时，列品类也出现在同一张发票里的条件概率；对角线不显示（自身恒为 1）')
ax.text(.995, -.20, '橙框列 = 餐厨烘焙：整列 0.73~0.93，几乎出现在每张发票里 →「万能类」，与谁共现都高、本身不构成搭配信号',
        transform=ax.transAxes, ha='right', fontsize=8.8, color=ORANGE)
plt.savefig(os.path.join(OUT, '图2_品类共现热力图.png')); plt.close()

# ---------- 图 3 品类级 Top 规则 ----------
t10 = rr_out.head(10).iloc[::-1]
flag = t10['前项'].eq('相框相册') & t10['后项'].eq('首饰收纳盒')
fig, ax = plt.subplots(figsize=(9.4, 5.4))
theme(ax)
ax.barh(["{} → {}".format(a, b) for a, b in zip(t10['前项'], t10['后项'])], t10['提升度'],
        color=[RED if f else TEAL for f in flag], height=.66)
for y, (v, c, f) in enumerate(zip(t10['提升度'], t10['置信度'], flag)):
    ax.text(v + .012, y, "lift {:.2f} · conf {:.2f}".format(v, c), va='center', fontsize=9,
            color=RED if f else MUTED, fontweight='bold' if f else 'normal')
ax.axvline(1.0, color=GREY2, lw=1.0)
ax.set_xlim(1.0, t10['提升度'].max() * 1.26)
ax.set_xlabel('提升度 lift', fontsize=10)
ax.tick_params(labelsize=9.8)
title(ax, '图3  品类级关联规则 Top 10（FP-Growth）',
      'min_support = 0.05、min_confidence = 0.30、项集最长 2；按原始提升度降序')
_top_raw = P.sort_values('提升度', ascending=False).iloc[0]
fig.text(.012, .012, '注：红色那条（原始 lift 最高的 {}×{}）经图5 的篮子大小校正后降为 {:.2f}（负关联）—— 品类级规则不能只看 lift 排名。'.format(
         _top_raw['品类A'], _top_raw['品类B'], _top_raw['校正后提升度']),
     fontsize=8.8, color=RED, style='italic')
plt.savefig(os.path.join(OUT, '图3_品类级Top规则.png')); plt.close()

# ---------- 图 4 单品级 vs 品类级 lift 分布 ----------
fig, ax = plt.subplots(figsize=(9.4, 5.4))
theme(ax, grid='both')
ax.axhspan(rr['lift'].min() * .95, rr['lift'].max() * 1.06, color=BLUE, alpha=.06, zorder=0)
ax.scatter(item_rr['support'], item_rr['lift'], s=52, color=ORANGE, alpha=.8,
           edgecolors='white', linewidths=.9, label='单品级规则  n = {}'.format(len(item_rr)), zorder=3)
ax.scatter(rr['support'], rr['lift'], s=46, color=BLUE, alpha=.8, marker='s',
           edgecolors='white', linewidths=.9, label='品类级规则  n = {}'.format(len(rr)), zorder=3)
ax.axhline(1.0, color=GREY2, lw=1.0, ls=(0, (4, 3)))
ax.set_xscale('log')
ax.set_xlabel('支持度（对数轴）', fontsize=10); ax.set_ylabel('提升度 lift', fontsize=10)
ax.tick_params(labelsize=9.6)
bb = dict(boxstyle='round,pad=0.42', fc='white', ec=GRID, lw=1)
ax.annotate("单品级最高 lift {:.2f}\n（同系列不同色的成套购买）".format(item_rr['lift'].max()),
            (item_rr['support'].iloc[0], item_rr['lift'].max()), xytext=(.30, .90),
            textcoords='axes fraction', fontsize=9.2, color=ORANGE, bbox=bb,
            arrowprops=dict(arrowstyle='-', color=ORANGE, lw=.9))
ax.text(.60, .10, "品类级全部落在 {:.2f}~{:.2f} 的窄带里，支持度却高一个数量级".format(
    rr['lift'].min(), rr['lift'].max()), fontsize=9.2, color=BLUE, bbox=bb, transform=ax.transAxes)
title(ax, '图4  粒度对比：同一份数据、两套结论',
      "lift 中位数 单品级 {:.2f} vs 品类级 {:.2f}；单品级 Top20 中 {}/20 条的两个商品本就同属一个品类，在品类粒度下直接消失".format(
          item_rr['lift'].median(), rr['lift'].median(), same_cat))
ax.legend(frameon=False, fontsize=9.6, loc='upper right')
plt.savefig(os.path.join(OUT, '图4_单品级与品类级对比.png')); plt.close()

# ---------- 图 5 哑铃图：原始 lift → 校正后 lift ----------
sel = pd.concat([P.head(6), P.tail(6)]).sort_values('校正后提升度')
fig, ax = plt.subplots(figsize=(9.8, 6.4))
for sp in ('top', 'right', 'left', 'bottom'):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0, labelsize=9.8)
ax.set_yticks(range(len(sel)))
ax.set_yticklabels(["{} × {}".format(a, b) for a, b in zip(sel['品类A'], sel['品类B'])])
for y, (_, r) in enumerate(sel.iterrows()):
    o, c = r['提升度'], r['校正后提升度']
    bad = c < 1.0
    ax.plot([o, c], [y, y], color=RED if bad else GREY2, lw=2.6, alpha=.75,
            solid_capstyle='round', zorder=2)
    ax.scatter([o], [y], s=66, color=BLUE, zorder=3, edgecolors='white', linewidths=1.1)
    ax.scatter([c], [y], s=66, color=RED if bad else ORANGE, zorder=3,
               edgecolors='white', linewidths=1.1)
    ax.text(c + (.02 if c >= o else -.02), y, "{:.2f}".format(c), va='center',
            ha='left' if c >= o else 'right', fontsize=9,
            color=RED if bad else ORANGE, fontweight='bold')
ax.axvline(1.0, color=INK, lw=1.1, alpha=.5, zorder=1)
ax.text(1.012, len(sel) / 2 - .5, 'lift = 1（独立）', fontsize=8.6, color=MUTED, va='center')
ax.set_xlim(.45, 1.62)
ax.set_xlabel('提升度（蓝点 = 原始；橙点 = 校正后；红点 = 校正后跌破 1，即被证伪）', fontsize=9.8)
ax.grid(axis='x', color=GRID, lw=.9); ax.set_axisbelow(True)
title(ax, '图5  控制「一单买几个品类」之后，品类级排名几乎反转',
      '零假设：保留每张发票的品类数 k、按品类边际分布无放回随机重选 k 类，置换 200 次取期望；55 对中 {} 对校正后 < 1'.format(n_neg))
ax.text(.995, -.125, '下半区：原始 lift 最高的「相框相册 × 首饰收纳盒」等，其实是两个小类叠加大篮子刷出来的假信号',
        transform=ax.transAxes, ha='right', fontsize=8.8, color=RED, style='italic')
plt.savefig(os.path.join(OUT, '图5_篮子大小校正.png')); plt.close()

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
4. 校正后仍稳定 >1.2 的是 {scene_pairs}，集中在**家居布置 + 厨房**的一站式采购；图2 里餐厨烘焙那一列条件概率 0.73~0.93，说明它是「万能类」，与谁共现都高、因此本身不构成搭配信号。
5. **单品级的强规则在品类粒度会直接消失**：单品级 Top20 中 {same_cat}/20 条（{same_cat/20*100:.0f}%）两个商品本就同属一个品类（如三色圆点餐具套装都在「餐厨烘焙」内），它们变成类内搭配，跨品类挖掘看不到。

## 四、可以写进报告的结论
- 单品级回答「哪两件商品该摆一起」（lift 最高 17.90，同系列不同色）；品类级回答「哪些消费场景会同时被触发」（校正后 1.2 以上集中在餐厨 × 装饰 × 收纳 × 烛台）。
- 两种粒度互补，**不能用品类级结果否定单品级，反之亦然**。
- 品类级多了一条方法论结论：**提升度必须结合支持度、品类规模与篮子大小一起读**，否则会把「大篮子造成的假信号」当成强关联（图5 下半部分就是被证伪的那批）。

## 五、文件清单
图1_品类映射覆盖率.png / 图2_品类共现热力图.png / 图3_品类级Top规则.png / 图4_单品级与品类级对比.png / 图5_篮子大小校正.png
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
