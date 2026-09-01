#!/usr/bin/env python3
"""
LLM-as-Judge 评审脚本
读取 test.py 生成的 generations_compare.json, 用强模型(如 Qwen-Max)对
微调前(base) / 微调后(finetuned) 的回复进行盲评打分:

  1. 每条样本: 随机打乱 base/finetuned 到 回复A/回复B (消除位置偏差)
  2. Judge 从 4 个维度打 1-5 分: 相关性 / 有帮助性 / 事实性 / 流畅性
  3. 选出 winner (盲评结束后还原真实身份)
  4. 汇总平均分 + 胜率, 保存 llm_judge_results.json

用法:
  # DashScope (阿里云百炼, 国内推荐, 默认 judge = qwen-max)
  set LLM_API_KEY=sk-xxx
  python llm_as_judge.py

  # OpenAI
  python llm_as_judge.py --api_base https://api.openai.com/v1 --judge_model gpt-4o

  # 指定某次测试的生成对照文件
  python llm_as_judge.py --input output/ecd_test_xxx/generations_compare.json
"""

import argparse
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """你是严格公正的AI回复质量评审。根据【用户问题】和【参考答案】, 评估两个AI回复的质量。

【用户问题】
{question}

【参考答案】
{reference}

【回复A】
{answer_a}

【回复B】
{answer_b}

从以下4个维度分别给两个回复打分(1-5分, 5分最好):
1. relevance(相关性): 是否切题回答了用户问题
2. helpfulness(有帮助性): 是否实际解决了用户问题
3. factuality(事实性): 内容是否与参考答案/常识一致, 有无编造
4. fluency(流畅性): 语言是否通顺自然、符合对话场景

严格按以下JSON格式输出, 不要输出任何其他内容:
{{"A": {{"relevance": 0, "helpfulness": 0, "factuality": 0, "fluency": 0}}, "B": {{"relevance": 0, "helpfulness": 0, "factuality": 0, "fluency": 0}}, "winner": "A", "reason": "一句话理由"}}"""


def find_latest_compare(output_root: str) -> Path:
    """自动找最新 test run 的 generations_compare.json"""
    candidates = sorted(Path(output_root).glob("*_test_*/generations_compare.json"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No generations_compare.json found under {output_root}, "
        f"run test.py first (and keep --max_gen_samples > 0)"
    )


def extract_json(text: str):
    """从 judge 回复中提取 JSON (容错: 代码块/前后杂文)"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def build_judge_client(args):
    """构建 OpenAI 兼容客户端"""
    from openai import OpenAI
    api_key = args.api_key
    if not api_key:
        raise SystemExit(
            "API key not set: use --api_key or environment variable LLM_API_KEY / OPENAI_API_KEY"
        )
    return OpenAI(api_key=api_key, base_url=args.api_base)


def judge_one(client, model, question, reference, answer_a, answer_b):
    """评审单条样本, 返回解析后的 dict 或 None"""
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference,
        answer_a=answer_a, answer_b=answer_b,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return extract_json(resp.choices[0].message.content or "")
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation")

    parser.add_argument("--input", type=str, default=None,
                       help="generations_compare.json 路径 (默认: 自动找最新 test 结果)")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="结果保存根目录")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="最多评审样本数")

    # Judge API 配置
    parser.add_argument("--api_base", type=str,
                       default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                       help="OpenAI 兼容 API 地址 (默认: 阿里云 DashScope)")
    parser.add_argument("--judge_model", type=str, default="qwen-max",
                       help="评审模型 (默认: qwen-max)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API Key (默认读环境变量 LLM_API_KEY / OPENAI_API_KEY)")

    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (A/B 顺序打乱)")

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # API key: 参数 > 环境变量
    import os
    if not args.api_key:
        args.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    # 定位输入
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_compare(args.output_dir)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    logger.info(f"Input: {input_path}")

    data = json.load(open(input_path, encoding='utf-8'))[:args.max_samples]
    logger.info(f"Judging {len(data)} samples with {args.judge_model} ...")

    client = build_judge_client(args)

    dims = ["relevance", "helpfulness", "factuality", "fluency"]
    scores = {"base": {d: [] for d in dims}, "finetuned": {d: [] for d in dims}}
    win = {"base": 0, "finetuned": 0, "tie": 0}
    details, failed = [], 0

    for i, item in enumerate(data):
        question = item["context"][-1]["content"] if item["context"] else ""
        reference = item["reference"]

        # 随机分配 A/B, 消除位置偏差
        if random.random() < 0.5:
            a_role, b_role = "base", "finetuned"
            answer_a, answer_b = item["base_output"], item["finetuned_output"]
        else:
            a_role, b_role = "finetuned", "base"
            answer_a, answer_b = item["finetuned_output"], item["base_output"]

        verdict = judge_one(client, args.judge_model,
                            question, reference, answer_a, answer_b)
        if verdict is None or "A" not in verdict or "B" not in verdict:
            failed += 1
            continue

        # 还原真实身份, 记分
        for role_key, side in [("base", "A" if a_role == "base" else "B"),
                               ("finetuned", "A" if a_role == "finetuned" else "B")]:
            for d in dims:
                v = verdict.get(side, {}).get(d)
                if isinstance(v, (int, float)):
                    scores[role_key][d].append(float(v))

        winner = verdict.get("winner", "tie")
        win[a_role if winner == "A" else b_role if winner == "B" else "tie"] += 1

        details.append({
            "question": question,
            "base_output": item["base_output"],
            "finetuned_output": item["finetuned_output"],
            "scores": {"base": verdict["A" if a_role == "base" else "B"],
                       "finetuned": verdict["A" if a_role == "finetuned" else "B"]},
            "winner": a_role if winner == "A" else b_role if winner == "B" else "tie",
            "reason": verdict.get("reason", ""),
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  judged {i + 1}/{len(data)}")

    # 汇总
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    n_valid = len(details)
    results = {
        "judge_model": args.judge_model,
        "num_samples": len(data),
        "num_valid": n_valid,
        "num_failed": failed,
        "avg_scores": {
            "base": {d: round(avg(scores["base"][d]), 3) for d in dims},
            "finetuned": {d: round(avg(scores["finetuned"][d]), 3) for d in dims},
        },
        "win_rate": {
            "base": round(win["base"] / n_valid, 3) if n_valid else 0,
            "finetuned": round(win["finetuned"] / n_valid, 3) if n_valid else 0,
            "tie": round(win["tie"] / n_valid, 3) if n_valid else 0,
        },
        "details": details,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # 保存到输入文件同目录
    out_path = input_path.parent / "llm_judge_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印摘要
    logger.info("=" * 72)
    logger.info(f"LLM-as-Judge 结果 (judge={args.judge_model}, 有效样本 {n_valid})")
    logger.info("=" * 72)
    logger.info(f"  {'维度':<14s}{'微调前':>10s}{'微调后':>10s}{'变化':>10s}")
    for d in dims:
        b, f_ = results["avg_scores"]["base"][d], results["avg_scores"]["finetuned"][d]
        logger.info(f"  {d:<14s}{b:>10.3f}{f_:>10.3f}{f_ - b:>+10.3f}")
    logger.info(f"  胜率: 微调后 {results['win_rate']['finetuned']*100:.1f}% | "
                f"微调前 {results['win_rate']['base']*100:.1f}% | "
                f"平局 {results['win_rate']['tie']*100:.1f}%")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()