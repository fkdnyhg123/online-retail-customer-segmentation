# -*- coding: utf-8 -*-
"""
生成报告 6.2 / 6.3 用的结果表 (CSV), 落到「算法工程师交付/」。

项目约定: 计算全在 utils/、界面全在 app.py。本脚本只做「调用 utils + 落盘」,
不含任何计算逻辑, 与历史脚本/week1_explore.py 同属非大屏脚本。
报告里的图不从这里出 —— 直接从大屏截图 (页面 5 的 📦 板块、页面 6 的 🌍 板块)。

用法:  .venv/Scripts/python.exe export_report_tables.py
"""
import os

import pandas as pd

from utils.data_cleaner import load_cleaned
from utils.category import (category_coverage, build_category_basket, mine_category_rules,
                            category_pair_table, sku_category_table)
from utils.association import prepare_basket_matrix, run_fpgrowth, country_rule_comparison

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, '算法工程师交付')
os.makedirs(OUT, exist_ok=True)

A, B = 'United Kingdom', 'Germany'

cln = load_cleaned(REPO)

# ---------- 6.2 品类级 ----------
cov = category_coverage(cln)
basket = build_category_basket(cln)
pairs = category_pair_table(basket)
cat_rules = mine_category_rules(basket, min_support=0.05, min_confidence=0.30, max_len=2)['rules']
cat_out = cat_rules.copy()
cat_out['前项'] = cat_out['antecedents'].map(lambda s: '+'.join(s))
cat_out['后项'] = cat_out['consequents'].map(lambda s: '+'.join(s))
cat_out = cat_out[['前项', '后项', 'support', 'confidence', 'lift']].rename(
    columns={'support': '支持度', 'confidence': '置信度', 'lift': '提升度'})

sku_category_table(cln).assign(Description=lambda d: d['Description'].astype(str)).to_csv(
    os.path.join(OUT, '表6.2-SKU品类映射表.csv'), index=False, encoding='utf-8-sig')
pairs.round(4).to_csv(os.path.join(OUT, '表6.2-品类对关联强度全表.csv'), index=False, encoding='utf-8-sig')
cat_out.round(4).to_csv(os.path.join(OUT, '表6.2-品类级关联规则.csv'), index=False, encoding='utf-8-sig')

# ---------- 6.3 跨国 ----------
g = (cln.groupby('Country')
     .agg(发票=('Invoice', 'nunique'), 客户=('Customer ID', 'nunique'), 收入=('Revenue', 'sum'))
     .sort_values('发票', ascending=False))
g['每单行数'] = (cln.groupby('Country').size() / g['发票']).round(1)
g['收入占比%'] = (g['收入'] / g['收入'].sum() * 100).round(2)
g.round(2).to_csv(os.path.join(OUT, '表6.3-各国可挖掘性.csv'), encoding='utf-8-sig')

cc = country_rule_comparison(cln, A, B, top_items=50, min_support=0.02, min_confidence=0.25)
cc['comparison'].round(4).to_csv(os.path.join(OUT, '表6.3-两国规则对比全表.csv'), index=False, encoding='utf-8-sig')
cc['comparison'][cc['comparison']['归属'] == '共通'].round(4).to_csv(
    os.path.join(OUT, '表6.3-共通规则.csv'), index=False, encoding='utf-8-sig')

print(f"6.2: 覆盖 {cov['coverage_pct']}% | 品类篮 {basket.shape[0]:,}×{basket.shape[1]} | "
      f"规则 {len(cat_out)} 条 | 校正后 <1 的品类对 {(pairs['校正后提升度'] < 1).sum()}/{len(pairs)}")
print(f"6.3: 共通 {cc['counts']['共通']} | {A} 特有 {cc['counts'][f'{A} 特有']} | {B} 特有 {cc['counts'][f'{B} 特有']}")
print("输出目录:", OUT)
for f in sorted(os.listdir(OUT)):
    print("   ", f)
