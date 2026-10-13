# -*- coding: utf-8 -*-
"""
6.3 跨国关联规则对比 —— 结果生成脚本
用法:  .venv/Scripts/python.exe 跨国规则对比/run_6p3_country_rules.py
产出:  图 4 张 + 3 个 CSV + 结果说明.md
依赖:  utils/association.py 的 country_rule_comparison()
"""
import os, sys, re, ast
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

from utils.association import country_rule_comparison

plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei'], 'axes.unicode_minus': False,
                     'figure.dpi': 200, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'savefig.bbox': 'tight', 'text.color': '#1F2933', 'axes.labelcolor': '#1F2933',
                     'xtick.color': '#7B8794', 'ytick.color': '#7B8794', 'axes.edgecolor': '#EDF0F3'})
INK, MUTED, GRID = '#1F2933', '#7B8794', '#EDF0F3'
BLUE, ORANGE, TEAL, RED, GREY2 = '#2C6FBB', '#E8833A', '#2A9D8F', '#C1444F', '#AAB7C4'

A, B = 'United Kingdom', 'Germany'
CN_NAME = {'United Kingdom': '英国', 'Germany': '德国', 'EIRE': '爱尔兰', 'Netherlands': '荷兰',
           'France': '法国', 'Denmark': '丹麦', 'Spain': '西班牙', 'Belgium': '比利时',
           'Sweden': '瑞典', 'Italy': '意大利', 'Portugal': '葡萄牙'}


def load_cleaned():
    _pq = os.path.join(REPO, '.cache_cleaned.parquet')
    if os.path.exists(_pq):
        return pd.read_parquet(_pq)
    from utils.data_cleaner import load_raw_data, clean_data
    return clean_data(load_raw_data(os.path.join(REPO, 'online_retail_II(1).xlsx')))


cln = load_cleaned()
src = open(os.path.join(REPO, 'app.py'), encoding='utf-8').read()
m = re.search(r'PRODUCT_CN\s*=\s*\{.*?\n\s*\}', src, re.S)
PRODUCT_CN = ast.literal_eval(m.group(0)[m.group(0).index('{'):])
DESC = cln.groupby('StockCode')['Description'].first()
CN_EXTRA = {
    'BAKING SET 9 PIECE RETROSPOT': '9件套复古圆点烘焙工具',
    'REGENCY CAKESTAND 3 TIER': '三层复古蛋糕架',
    'RETRO SPOT CAKE STAND': '复古圆点蛋糕架',
    'RETRO SPOT TEA SET': '复古圆点茶具套装',
    'SET/5 RED SPOTTY LID GLASS BOWLS': '5件套红圆点带盖玻璃碗',
    'SET OF 72 RETRO SPOT PAPER DOILIES': '72张复古圆点纸垫',
    'PACK OF 20 RED SPOTTY PAPER NAPKINS': '20张红圆点纸餐巾',
}


def cn(code):
    d = str(DESC.get(code, code))
    return PRODUCT_CN.get(d.strip().upper()) or CN_EXTRA.get(d.strip().upper(), d[:16])


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


# ---------- 各国可挖掘性 ----------
g = (cln.groupby('Country')
     .agg(发票=('Invoice', 'nunique'), 客户=('Customer ID', 'nunique'), 收入=('Revenue', 'sum'))
     .sort_values('发票', ascending=False))
g['每单行数'] = (cln.groupby('Country').size() / g['发票']).round(1)
g['收入占比%'] = (g['收入'] / g['收入'].sum() * 100).round(2)

# ---------- 跨国对比 ----------
res = country_rule_comparison(cln, A, B, top_items=50, min_support=0.02, min_confidence=0.25)
comp = res['comparison']
common = comp[comp['归属'] == '共通'].sort_values(f'{A} 提升度', ascending=False)
only_a = comp[comp['归属'] == f'{A} 特有'].sort_values(f'{A} 提升度', ascending=False)
only_b = comp[comp['归属'] == f'{B} 特有'].sort_values(f'{B} 提升度', ascending=False)
st = res['stats']

# ---------- 图 1 国家选择依据 ----------
top8 = g.head(8).iloc[::-1]
labels = [CN_NAME.get(k, k) for k in top8.index]
fig, ax = plt.subplots(figsize=(9.4, 4.8))
theme(ax)
cols = [RED if k == 'EIRE' else (BLUE if k in (A, B) else GREY2) for k in top8.index]
ax.barh(labels, top8['发票'], color=cols, height=.66)
for i, (k, v) in enumerate(zip(top8.index, top8['发票'])):
    ax.text(v * 1.15, i, f"{v:,} 张发票 · {int(g.loc[k, '客户']):,} 位客户 · 收入占 {g.loc[k, '收入占比%']:.1f}%",
            va='center', fontsize=9, color=INK if k in (A, B, 'EIRE') else MUTED)
ax.set_xscale('log')
ax.set_xlabel('发票数（对数轴）', fontsize=10)
ax.tick_params(labelsize=10)
ax.set_xlim(1, top8['发票'].max() * 40)
title(ax, '图1  Top2 国家选择依据：按发票数与客户数，不按收入',
      '爱尔兰收入占 3.88% 排第二，但只有 5 位客户、平均每单 29 行 —— 是批发账户不是市场，排除；'
      '德国 328 张发票 / 67 位客户，是海外统计意义最强的市场')
plt.savefig(os.path.join(OUT, '图1_国家选择依据.png')); plt.close()

# ---------- 图 2 共通规则 lift 对比 ----------
fig, ax = plt.subplots(figsize=(9.8, 5.6))
theme(ax)
y = np.arange(len(common))[::-1]
h = .34
ax.barh(y + h / 2, common[f'{A} 提升度'], height=h, color=BLUE, label=f'{CN_NAME[A]} lift')
ax.barh(y - h / 2, common[f'{B} 提升度'], height=h, color=ORANGE, label=f'{CN_NAME[B]} lift')
ax.set_yticks(y)
ax.set_yticklabels([f"{cn(a)} × {cn(b)}" for a, b in zip(common['商品A'], common['商品B'])], fontsize=9.4)
for yy, (_, r) in zip(y, common.iterrows()):
    ax.text(r[f'{A} 提升度'] + .12, yy + h / 2, f"{r[f'{A} 提升度']:.2f} (n={int(r[f'{A} 共现笔数'])})",
            va='center', fontsize=8.6, color=BLUE)
    ax.text(r[f'{B} 提升度'] + .12, yy - h / 2, f"{r[f'{B} 提升度']:.2f} (n={int(r[f'{B} 共现笔数'])})",
            va='center', fontsize=8.6, color=ORANGE)
ax.axvline(1.0, color=GREY2, lw=1)
ax.set_xlabel('提升度 lift', fontsize=10)
ax.set_xlim(0, common[f'{A} 提升度'].max() * 1.55)
ax.legend(frameon=False, fontsize=9.6, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2)
ax.margins(y=0.02)
fig.subplots_adjust(bottom=0.20)
title(ax, f'图2  {len(common)} 条共通规则：同一条搭配在两国都成立',
      f'固定全局 Top-50 商品全集、min_support=0.02、min_confidence=0.25、项集最长 2；'
      f'n 为共现的绝对笔数 —— 德国侧只有 {int(common[f"{B} 共现笔数"].min())}~{int(common[f"{B} 共现笔数"].max())} 笔，属方向性信号')
plt.savefig(os.path.join(OUT, '图2_共通规则对比.png')); plt.close()

# ---------- 图 3 特有规则 ----------
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=False)
for ax, df_, c, col, lab in ((axes[0], only_a.head(8), A, BLUE, CN_NAME[A]),
                             (axes[1], only_b.head(8), B, ORANGE, CN_NAME[B])):
    theme(ax)
    yy = np.arange(len(df_))[::-1]
    ax.barh(yy, df_[f'{c} 提升度'], color=col, height=.64)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{cn(a)} × {cn(b)}" for a, b in zip(df_['商品A'], df_['商品B'])], fontsize=8.8)
    for yv, (_, r) in zip(yy, df_.iterrows()):
        ax.text(r[f'{c} 提升度'] + .1, yv, f"{r[f'{c} 提升度']:.2f} (n={int(r[f'{c} 共现笔数'])})",
                va='center', fontsize=8.4, color=MUTED)
    ax.axvline(1.0, color=GREY2, lw=1)
    ax.set_xlim(0, df_[f'{c} 提升度'].max() * 1.5)
    ax.set_xlabel('提升度 lift', fontsize=9.6)
    title(ax, f'{lab}特有 Top 8（共 {len(only_a) if c == A else len(only_b)} 条）')
fig.suptitle('图3  特有规则：两国各自的本地搭配', fontsize=12.5, fontweight='bold', color=INK,
             x=.02, ha='left', y=1.06)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '图3_特有规则.png')); plt.close()

# ---------- 图 4 证据量不对称 ----------
fig, ax = plt.subplots(figsize=(9.2, 4.8))
theme(ax, grid='both')
ax.scatter(common[f'{A} 共现笔数'], common[f'{A} 提升度'], s=70, color=BLUE,
           edgecolors='white', linewidths=1, label=f'{CN_NAME[A]}（主场市场）', zorder=3)
ax.scatter(common[f'{B} 共现笔数'], common[f'{B} 提升度'], s=70, color=ORANGE, marker='s',
           edgecolors='white', linewidths=1, label=f'{CN_NAME[B]}（海外市场）', zorder=3)
ax.set_xscale('log')
ax.set_xlabel('共现的绝对笔数 n（对数轴）', fontsize=10)
ax.set_ylabel('提升度 lift', fontsize=10)
ax.tick_params(labelsize=9.6)
ax.legend(frameon=False, fontsize=9.6)
title(ax, '图4  同一条共通规则，两国的证据量差两个数量级',
      f'英国侧 n = {int(common[f"{A} 共现笔数"].min())}~{int(common[f"{A} 共现笔数"].max())} 笔是硬证据；'
      f'德国侧 n = {int(common[f"{B} 共现笔数"].min())}~{int(common[f"{B} 共现笔数"].max())} 笔，'
      'lift 的方向可信、数值不可信 —— 报告里必须这样写')
plt.savefig(os.path.join(OUT, '图4_证据量不对称.png')); plt.close()

# ---------- CSV / 说明 ----------
comp.round(4).to_csv(os.path.join(OUT, '表_两国规则对比全表.csv'), index=False, encoding='utf-8-sig')
common.round(4).to_csv(os.path.join(OUT, '表_共通规则.csv'), index=False, encoding='utf-8-sig')
g.round(2).to_csv(os.path.join(OUT, '表_各国可挖掘性.csv'), encoding='utf-8-sig')

top_common = common.iloc[0]
md = f"""# 6.3 跨国关联规则对比 —— 结果说明

## 一、Top2 国家选择依据（对应 6.3.1）
- 选 **{CN_NAME[A]}（{g.loc[A,'发票']:,} 张发票 / {g.loc[A,'客户']:,} 位客户）** 与 **{CN_NAME[B]}（{g.loc[B,'发票']:,} 张 / {g.loc[B,'客户']:,} 位）**。
- **不按收入选**：{CN_NAME['EIRE']} 收入占 {g.loc['EIRE','收入占比%']:.2f}% 排第二，但只有 {int(g.loc['EIRE','客户'])} 位客户、平均每单 {g.loc['EIRE','每单行数']:.0f} 行，是批发账户不是市场，其「规则」只是 5 个人的采购习惯。
- 也不按发票数硬排第二名：{CN_NAME['Netherlands']} 只有 {int(g.loc['Netherlands','发票'])} 张发票 / {int(g.loc['Netherlands','客户'])} 位客户，2% 支持度只对应 {int(g.loc['Netherlands','发票']*0.02)} 笔共现，统计意义不足。

## 二、可比性口径（必须写进报告）
1. **固定商品全集**：两国共用全局 Top-50 高频 StockCode 作为列集合；若各取本国 Top-N，「共通/特有」会变成商品集合差异而非搭配差异。
2. **相对支持度 + 绝对笔数并报**：min_support=0.02 与单品级口径一致，但德国的 2% 只有 {int(st[B]['交易数']*0.02)} 笔，所以每条规则同时给出共现笔数 n。
3. **A→B 与 B→A 视为同一条**：按无序商品对取集合交/差。
4. 阈值 min_confidence=0.25、lift ≥ 1、项集最长 2（小国样本下 max_len=3 会组合爆炸）。

## 三、结果
- 参与交易：{CN_NAME[A]} {st[A]['交易数']:,} 笔（含 ≥2 件目标商品 {st[A]['含≥2件目标商品']:,}）、{CN_NAME[B]} {st[B]['交易数']} 笔（{st[B]['含≥2件目标商品']}）。
- 规则数：{CN_NAME[A]} {st[A]['规则数']} 条、{CN_NAME[B]} {st[B]['规则数']} 条。
- **共通 {res['counts']['共通']} 条 / {CN_NAME[A]}特有 {res['counts'][f'{A} 特有']} 条 / {CN_NAME[B]}特有 {res['counts'][f'{B} 特有']} 条**。
- 共通规则集中在两类：**午餐袋各花色两两共购**（WOODLAND / CARS BLUE / RED SPOTTY / SPACEBOY / SUKI / PINK RETROSPOT）与**陶瓷首饰盒两款共购**（草莓 ↔ 甜心），即「同系列不同花色」的搭配行为跨市场稳定，与 6.1 单品级结论互相印证。
- 最强共通规则：{cn(top_common['商品A'])} × {cn(top_common['商品B'])}，{CN_NAME[A]} lift {top_common[f'{A} 提升度']:.2f}（n={int(top_common[f'{A} 共现笔数'])}）、{CN_NAME[B]} lift {top_common[f'{B} 提升度']:.2f}（n={int(top_common[f'{B} 共现笔数'])}）。

## 四、必须坦白的局限（答辩重点）
- 证据量不对称：同一条共通规则，{CN_NAME[A]}侧 n 为 {int(common[f'{A} 共现笔数'].min())}~{int(common[f'{A} 共现笔数'].max())} 笔，{CN_NAME[B]}侧只有 {int(common[f'{B} 共现笔数'].min())}~{int(common[f'{B} 共现笔数'].max())} 笔，**差两个数量级**。德国侧的 lift 只能当方向性信号，数值不可信；「共通」应表述为「在两个市场同向出现」，不是「同样强度的规律」。
- 德国 213 笔交易里 2% 支持度 = 4 笔共现，任何一条「特有规则」都可能只是几笔大单的偶然，所以特有规则只作展示、不作决策依据。
- 若老师追问「为什么不做爱尔兰」：5 位客户的样本做规则挖掘，等于把 5 个人的购物车当国家偏好。

## 五、文件清单
图1_国家选择依据.png / 图2_共通规则对比.png / 图3_特有规则.png / 图4_证据量不对称.png
表_两国规则对比全表.csv / 表_共通规则.csv / 表_各国可挖掘性.csv
"""
open(os.path.join(OUT, '结果说明.md'), 'w', encoding='utf-8').write(md)

print(f"共通 {res['counts']['共通']} | {CN_NAME[A]}特有 {res['counts'][f'{A} 特有']} | {CN_NAME[B]}特有 {res['counts'][f'{B} 特有']}")
print(f"交易 {st[A]['交易数']:,} / {st[B]['交易数']} | 规则 {st[A]['规则数']} / {st[B]['规则数']}")
print("输出目录:", OUT)
for f in sorted(os.listdir(OUT)):
    print("   ", f)
