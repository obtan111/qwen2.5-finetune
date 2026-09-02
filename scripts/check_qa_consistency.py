#!/usr/bin/env python3
"""
QA 答案一致性(幂等性)检测
ECD 是多店铺混合语料, 同一表面问题在不同店铺有不同正确答案(快递/克重/价格等)。
本脚本量化每类事实型问题的答案集中度:

  一致性得分 = 最高频答案占比 (top1 / total)
  >= 0.90  => 答案全局唯一, 模型可安全学习
  <  0.90  => 答案因店而异, 学习会导致模型胡乱拼凑, 应剔除

用法:
  python scripts/check_qa_consistency.py                    # 默认检查 ecd_food/train.jsonl
  python scripts/check_qa_consistency.py --input 其他.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

COURIERS = ['韵达', '邮政', 'EMS', '顺丰', '圆通', '中通', '申通', '百世', '天天', '汇通', '京东物流']


def extract_courier(reply: str):
    found = tuple(sorted({c for c in COURIERS if c in reply}))
    return ' + '.join(found) if found else None


def extract_grams(reply: str):
    found = tuple(sorted(set(re.findall(r'\d+(?:\.\d+)?[克gG]', reply))))
    return ', '.join(found) if found else None


def extract_days(reply: str):
    found = tuple(sorted(set(re.findall(r'\d+\s*天|\d+\s*日|48小时|\d+\s*小时', reply))))
    return ', '.join(found) if found else None


def extract_price(reply: str):
    found = tuple(sorted(set(re.findall(r'\d+(?:\.\d+)?元|\d+(?:\.\d+)?块|满\d+', reply))))
    return ', '.join(found) if found else None


# 事实型问题模式: (问题名, 问题匹配正则, 答案提取器)
QUESTION_PATTERNS = [
    ("什么快递",     re.compile(r'什么快递|哪家快递|发什么|哪个快递'), extract_courier),
    ("多少克/多重",  re.compile(r'多少克|几克|多重|净重|一袋多少|一包多少'), extract_grams),
    ("几天到/多久",  re.compile(r'几天到|多久到|几天能到|什么时候到|多长时间'), extract_days),
    ("多少钱/价格",  re.compile(r'多少钱|价格|贵不贵|多少钱一|卖多少'), extract_price),
    ("什么时候发货", re.compile(r'什么时候发货|何时发货|几天发|什么时候发'), extract_days),
]


def main():
    parser = argparse.ArgumentParser(description="QA 答案一致性检测")
    parser.add_argument("--input", type=str,
                        default=str(Path(__file__).resolve().parent.parent
                                    / "data" / "ecd_food" / "train.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="一致性阈值 (默认0.90)")
    args = parser.parse_args()

    data = [json.loads(l) for l in open(args.input, encoding='utf-8')]
    print(f"数据集: {args.input} ({len(data)} 条)\n")

    # 对每类问题: 收集 (问题->答案) 样本
    results = []
    for qname, pat, extractor in QUESTION_PATTERNS:
        answers = Counter()
        for it in data:
            msgs = it['messages']
            # 会话里任一 user 轮命中问题模式 -> 取最后客服回复提取答案
            if any(m['role'] == 'user' and pat.search(m['content']) for m in msgs[1:-1]):
                ans = extractor(msgs[-1]['content'])
                if ans:
                    answers[ans] += 1
        total = sum(answers.values())
        if total == 0:
            continue
        top_ans, top_n = answers.most_common(1)[0]
        consistency = top_n / total
        verdict = ("✅ 一致, 可学习" if consistency >= args.threshold
                   else "❌ 因店而异, 应剔除")
        results.append((qname, total, len(answers), top_ans, consistency, verdict))

    print(f"{'问题类型':<12s}{'样本数':>7s}{'答案种数':>8s}{'最高频答案':<26s}{'占比':>7s}  结论")
    print("-" * 80)
    for qname, total, n_ans, top_ans, cons, verdict in results:
        print(f"{qname:<12s}{total:>7d}{n_ans:>8d}  {top_ans[:24]:<26s}{cons*100:>6.1f}%  {verdict}")
    print("-" * 80)
    print(f"判定标准: 最高频答案占比 >= {args.threshold:.0%} 视为一致(可学习), 否则剔除")
    print("\n说明: 答案因店而异的问题, 模型无法从对话上下文判断该用哪个答案,")
    print("      学习后会在生成时随机混搭多家店铺话术 (自相矛盾)。")


if __name__ == "__main__":
    main()