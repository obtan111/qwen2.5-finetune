#!/usr/bin/env python3
"""
将 ECD 电商对话语料 (E-commerce Dialogue Corpus) 转换为 ChatML JSONL 格式

数据原始格式(每行): label \t 对话轮次(以\\t分隔) \t 候选回复
  - label=1: 正样本(正确的回复)
  - label=0: 负样本(干扰回复, 对 SFT 无用, 自动跳过)
  - 对话轮次: 奇数位=顾客(user), 偶数位=客服(assistant)
  - 每行最后一列 = 该会话的正确回复(assistant)

用法示例:
  # 直接运行(默认: 从 train.txt 抽取 30000 条 -> data/ecd/train.jsonl)
  python scripts/convert_ecd_to_chatml.py

  # 转换 dev 集为 eval 数据
  python scripts/convert_ecd_to_chatml.py \
      --input "dataset/E-commerce dataset/dev.txt" \
      --output data/ecd/eval.jsonl

  # 全量转换(约 50 万条, 慎用)
  python scripts/convert_ecd_to_chatml.py --max_samples 1000000
"""

import argparse
import json
import random
import re
from pathlib import Path

from analyze_ecd_categories import CATEGORIES

# 事实型 QA 问题模式: 答案因店而异(快递/克重/价格/时效), 学习会导致模型随机拼凑答案
FACT_QA_PATTERNS = [
    re.compile(r'什么快递|哪家快递|发什么|哪个快递'),
    re.compile(r'多少克|几克|多重|净重|一袋多少|一包多少'),
    re.compile(r'几天到|多久到|几天能到|什么时候到|多长时间'),
    re.compile(r'多少钱|价格|贵不贵|多少钱一|卖多少'),
    re.compile(r'什么时候发货|何时发货|几天发|什么时候发'),
]


def is_fact_qa(messages) -> bool:
    """会话中顾客问了事实型问题 -> True (答案因店而异, 应剔除)"""
    for m in messages:
        if m['role'] == 'user' and any(p.search(m['content']) for p in FACT_QA_PATTERNS):
            return True
    return False

SYSTEM_PROMPT = "你是一个专业的电商客服，负责解答顾客关于商品、下单、快递、发货、退换货、优惠等问题。"

# 基于脚本位置解析项目根目录, 无论从哪个目录运行都能找到默认文件
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "E-commerce dataset" / "train.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ecd" / "train.jsonl"


def cleanup(text: str) -> str:
    """去掉分词产生的空格, 恢复为连续中文文本"""
    return text.replace(' ', '').replace('\u3000', '').strip()


def is_clean_reply(reply: str) -> bool:
    """回复质量检查: 过滤大杂烩模板/乱码/数字串样本"""
    if not reply:
        return False
    # 连续数字/字母长串 (ID串/乱码, 如 1020102610251022... / ctype / acket)
    if re.search(r"[0-9A-Za-z]{12,}", reply):
        return False
    # 中文字符占比过低 (英文乱码模板)
    cn = sum(1 for c in reply if '\u4e00' <= c <= '\u9fff')
    if cn / max(len(reply), 1) < 0.4:
        return False
    return True


def process_row(parts, min_turns, max_turns, add_system, max_reply_len):
    """将一行 ECD 数据转换为 ChatML 格式, 无效则返回 None"""
    if len(parts) < 3:
        return None

    session = parts[1:-1]   # 对话轮次(奇数位=顾客, 偶数位=客服)
    response = parts[-1]    # 候选回复(正样本 = 正确回复)

    if not response.strip():
        return None
    if any(not s.strip() for s in session):
        return None  # 存在空轮次, 角色交替规律可能被破坏, 跳过

    turns = len(session)
    if turns < min_turns or turns > max_turns:
        return None

    # 回复质量过滤: 短而聚焦的客服回复才是好样本
    reply = cleanup(response)
    if len(reply) > max_reply_len:
        return None  # 大杂烩长模板
    if not is_clean_reply(reply):
        return None  # 乱码/数字串/英文模板

    messages = []
    if add_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    for i, utt in enumerate(session):
        role = 'user' if i % 2 == 0 else 'assistant'
        messages.append({"role": role, "content": cleanup(utt)})

    messages.append({"role": "assistant", "content": reply})
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(
        description="Convert ECD corpus to ChatML JSONL"
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="ECD 输入文件 (默认: ECD train.txt)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="输出 JSONL 文件路径 (默认: 项目 data/train.jsonl)")
    parser.add_argument("--max_samples", type=int, default=30000,
                        help="随机抽取最多多少条正样本 (默认30000; 全量请传一个大数如 --max_samples 1000000)")
    parser.add_argument("--min_turns", type=int, default=1,
                        help="最小对话轮次(默认1, 单轮问答也保留)")
    parser.add_argument("--max_turns", type=int, default=30,
                        help="最大对话轮次(默认30, 过滤超长会话)")
    parser.add_argument("--max_reply_len", type=int, default=60,
                        help="回复最大字符数(默认60, 过滤大杂烩长模板/乱码/数字串)")
    parser.add_argument("--no_system", action="store_true",
                        help="不加系统提示")
    parser.add_argument("--category", type=str, default=None,
                        choices=list(CATEGORIES.keys()),
                        help="只保留指定品类的对话 (如 '食品/零食')")
    parser.add_argument("--drop_fact_qa", action="store_true", default=False,
                        help="剔除事实型QA(快递/克重/价格/时效等答案因店而异的问题)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    add_system = not args.no_system
    max_n = args.max_samples

    skipped_label = 0   # label=0 的负样本
    skipped_invalid = 0  # 格式异常/轮次/质量过滤掉的
    skipped_category = 0  # 不属于指定品类的
    skipped_fact_qa = 0   # 事实型QA(答案因店而异)
    n_seen = 0           # 已见正样本计数(用于 reservoir)
    pool = []            # reservoir 采样池

    with open(in_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')

            if len(parts) < 3:
                skipped_invalid += 1
                continue
            if parts[0] != '1':
                skipped_label += 1
                continue

            item = process_row(parts, args.min_turns, args.max_turns,
                               add_system, args.max_reply_len)
            if item is None:
                skipped_invalid += 1
                continue

            # 品类过滤: 只保留命中指定品类关键词的对话
            if args.category:
                text = ''.join(parts[1:]).replace(' ', '')
                if not any(w in text for w in CATEGORIES[args.category]):
                    skipped_category += 1
                    continue

            # 事实型QA剔除: 答案因店而异, 模型无法判断该用哪个答案
            if args.drop_fact_qa and is_fact_qa(item['messages']):
                skipped_fact_qa += 1
                continue

            n_seen += 1
            if max_n is None:
                pool.append(item)
            else:
                # reservoir sampling: 等概率随机抽取 max_n 条, 内存占用恒定
                if len(pool) < max_n:
                    pool.append(item)
                else:
                    j = random.randrange(n_seen)
                    if j < max_n:
                        pool[j] = item

    with open(out_path, 'w', encoding='utf-8') as out:
        for item in pool:
            out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[done] input={in_path}")
    print(f"  converted={len(pool)} | skipped(label=0)={skipped_label} "
          f"| skipped(invalid)={skipped_invalid} | skipped(其他品类)={skipped_category} "
          f"| skipped(事实型QA)={skipped_fact_qa}")
    print(f"  saved={out_path}")


if __name__ == "__main__":
    main()