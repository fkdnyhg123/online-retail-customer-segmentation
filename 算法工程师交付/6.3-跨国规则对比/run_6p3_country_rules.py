# -*- coding: utf-8 -*-
"""
6.3 跨国关联规则对比 —— 结果生成脚本
用法:  .venv/Scripts/python.exe 跨国规则对比/run_6p3_country_rules.py
产出:  图 4 张 + 3 个 CSV + 结果说明.md
依赖:  utils/association.py 的 country_rule_comparison()
"""
import os, sys, re, ast
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

from utils.association import country_rule_comparison


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
图不再生成静态文件：报告用图直接从大屏页面 6「🌍 跨国关联规则对比 (国家粒度)」板块截图
（共通规则两国 lift 并排 / 特有规则双栏 / 证据量散点三张；国家选择依据可用页面 6 的国家明细表截图代替）。
表_两国规则对比全表.csv / 表_共通规则.csv / 表_各国可挖掘性.csv
"""
open(os.path.join(OUT, '结果说明.md'), 'w', encoding='utf-8').write(md)

print(f"共通 {res['counts']['共通']} | {CN_NAME[A]}特有 {res['counts'][f'{A} 特有']} | {CN_NAME[B]}特有 {res['counts'][f'{B} 特有']}")
print(f"交易 {st[A]['交易数']:,} / {st[B]['交易数']} | 规则 {st[A]['规则数']} / {st[B]['规则数']}")
print("输出目录:", OUT)
for f in sorted(os.listdir(OUT)):
    print("   ", f)
