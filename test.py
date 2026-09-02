#!/usr/bin/env python3
"""
Qwen 2.5 微调测试脚本
1. Loss 指标评估: 微调前后模型在测试集上的 eval_loss / token accuracy
2. 生成质量评估: 两模型实际生成回复, 计算 Exact Match / BLEU / ROUGE, 保存对照样本
结果保存到 output/{数据集名}_test_{时间戳}/

用法:
  python test.py                                        # 自动找最新训练结果 + 对比基座
  python test.py --test_data_path ./data/ecd/test.jsonl
  python test.py --max_gen_samples 200                  # 生成评估样本数(默认100)
  python test.py --skip_base                            # 只测微调后
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig

from train import load_and_preprocess_data, get_dataset_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------- 基础工具 ----------

def find_latest_model(output_root: str) -> str:
    """在 output 根目录下找最新训练 run 的 final 模型"""
    candidates = sorted(Path(output_root).glob("*_train_*/final"))
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError(
        f"No trained model found under {output_root}, run train.py first"
    )


def resolve_base_path(base_path: str):
    """解析基座模型路径 (adapter 记录的可能是相对路径, 做兜底解析)"""
    p = Path(base_path)
    if p.exists():
        return str(p)
    project_root = Path(__file__).resolve().parent
    p2 = project_root / base_path
    if p2.exists():
        return str(p2)
    return None


def load_model(model_path: str, args):
    """加载模型 (自动识别 LoRA adapter 或完整模型)"""
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if (Path(model_path) / "adapter_config.json").exists():
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    return model


def load_raw_samples(data_path: str, max_samples: int):
    """读取原始 messages 样本 (用于生成评估), 统一为 messages 格式"""
    p = Path(data_path)
    samples = []
    if p.suffix == '.jsonl':
        for line in open(p, encoding='utf-8'):
            if line.strip():
                samples.append(json.loads(line))
    else:
        samples = json.load(open(p, encoding='utf-8'))

    msgs = []
    for item in samples[:max_samples]:
        if 'messages' in item:
            msgs.append(item['messages'])
        else:  # prompt/reference 格式包装
            msgs.append([
                {"role": "user", "content": item.get('prompt', item.get('instruction', ''))},
                {"role": "assistant",
                 "content": item.get('reference', item.get('answer', item.get('output', '')))},
            ])
    return msgs


# ---------- Loss 指标评估 ----------

def evaluate_model(model, tokenizer, args, run_dir: Path):
    """构建评估 trainer 并返回 loss 类指标"""
    test_dataset = load_and_preprocess_data(
        args.test_data_path, tokenizer, args.max_seq_length
    )['train']

    eval_args = SFTConfig(
        output_dir=str(run_dir),
        per_device_eval_batch_size=args.batch_size,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=eval_args,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    return trainer.evaluate()


# ---------- 生成质量评估 ----------

def generate_responses(model, tokenizer, samples, args):
    """逐条生成回复 (messages[:-1] 为上下文), 并统计性能指标:
    TTFT(首token延迟) / prefill吞吐(输入) / decode吞吐(输出)"""
    model.eval()
    outputs, perf = [], []
    for i, messages in enumerate(samples):
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_in = inputs['input_ids'].shape[1]

        # 1) 单独测首 token 延迟 (prefill + 1 步 decode)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=1, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        ttft = time.perf_counter() - t0

        # 2) 完整生成
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,           # greedy, 保证可复现
                pad_token_id=tokenizer.pad_token_id,
            )
        total = time.perf_counter() - t0
        n_out = out.shape[1] - n_in

        text = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        outputs.append(text.strip())
        perf.append({
            "input_tokens": n_in,
            "output_tokens": n_out,
            "ttft_ms": ttft * 1000,
            "prefill_tokens_per_sec": n_in / ttft if ttft > 0 else 0.0,
            "decode_tokens_per_sec": max(n_out - 1, 1) / max(total - ttft, 1e-6),
            "total_time_sec": total,
        })
        if (i + 1) % 20 == 0:
            logger.info(f"  generated {i + 1}/{len(samples)}")
    return outputs, perf


def compute_generation_metrics(references, hypotheses):
    """计算生成指标: exact_match / char-BLEU / ROUGE (中文按字符切分)"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

    n = max(len(references), 1)
    exact = sum(1 for r, h in zip(references, hypotheses)
                if r.strip() == h.strip())
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    smooth = SmoothingFunction().method1
    bleus, r1s, rl_s = [], [], []
    for r, h in zip(references, hypotheses):
        rc, hc = list(r.strip()), list(h.strip())
        try:
            bleus.append(sentence_bleu([rc], hc, smoothing_function=smooth))
        except Exception:
            bleus.append(0.0)
        s = scorer.score(' '.join(rc), ' '.join(hc))
        r1s.append(s['rouge1'].fmeasure)
        rl_s.append(s['rougeL'].fmeasure)
    return {
        "num_samples": len(references),
        "exact_match": exact / n,
        "bleu": sum(bleus) / n,
        "rouge1": sum(r1s) / n,
        "rougeL": sum(rl_s) / n,
    }


def generation_eval(model, tokenizer, samples, args):
    """生成 + 质量指标 + 性能指标, 返回 (metrics, outputs)"""
    outputs, perf = generate_responses(model, tokenizer, samples, args)
    references = [m[-1]['content'] for m in samples]
    metrics = compute_generation_metrics(references, outputs)

    # 汇总性能指标 (平铺, 便于对比表直接展示)
    if perf:
        n = len(perf)
        metrics["ttft_ms"] = sum(p["ttft_ms"] for p in perf) / n
        metrics["prefill_tokens_per_sec"] = sum(p["prefill_tokens_per_sec"] for p in perf) / n
        metrics["decode_tokens_per_sec"] = sum(p["decode_tokens_per_sec"] for p in perf) / n
        metrics["avg_input_tokens"] = sum(p["input_tokens"] for p in perf) / n
        metrics["avg_output_tokens"] = sum(p["output_tokens"] for p in perf) / n
    return metrics, outputs


# ---------- 参数 ----------

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen 2.5 Fine-tuning Test")

    parser.add_argument("--model_path", type=str, default=None,
                       help="训练好的模型路径 (默认: 自动找 output 下最新训练结果)")
    parser.add_argument("--test_data_path", type=str, default="./data/ecd/test.jsonl",
                       help="测试数据路径 (jsonl/json 格式)")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="结果保存根目录")
    parser.add_argument("--skip_base", action="store_true", default=False,
                       help="跳过基座模型对比, 只评估微调后模型")

    # 生成评估参数
    parser.add_argument("--max_gen_samples", type=int, default=100,
                       help="生成质量评估的样本数 (0=关闭, 默认100)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="生成回复的最大新 token 数")

    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="评估批大小")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="是否使用 FP16")
    parser.add_argument("--bf16", action="store_true", default=False,
                       help="是否使用 BF16")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    return parser.parse_args()


# ---------- 主流程 ----------

def test(args):
    """测试主函数: loss 指标 + 生成质量, 微调前后对比"""
    logger.info("Starting testing...")

    if args.model_path is None:
        args.model_path = find_latest_model(args.output_dir)
    logger.info(f"Model to test: {args.model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = get_dataset_name(args.test_data_path)
    run_dir = Path(args.output_dir) / f"{dataset_name}_test_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 生成评估样本
    gen_samples = load_raw_samples(args.test_data_path, args.max_gen_samples) \
        if args.max_gen_samples > 0 else []

    is_adapter = (Path(args.model_path) / "adapter_config.json").exists()
    results = {}

    base_path = None
    if is_adapter and not args.skip_base:
        adapter_cfg = json.load(open(Path(args.model_path) / "adapter_config.json",
                                     encoding='utf-8'))
        base_path = resolve_base_path(adapter_cfg.get("base_model_name_or_path", ""))

    if base_path:
        # ================= 微调前后对比模式 =================
        # [1/2] 基座模型 (微调前): loss 指标 + 生成
        logger.info(f"[1/2] Evaluating BASE model: {base_path}")
        base_model = load_model(base_path, args)
        results["base_model"] = evaluate_model(base_model, tokenizer, args, run_dir)
        if gen_samples:
            logger.info(f"[1/2] Generation evaluation ({len(gen_samples)} samples)...")
            results["base_generation"], base_outputs = \
                generation_eval(base_model, tokenizer, gen_samples, args)
        else:
            base_outputs = None
        del base_model
        torch.cuda.empty_cache()

        # [2/2] 微调后模型: loss 指标 + 生成
        logger.info(f"[2/2] Evaluating FINETUNED model: {args.model_path}")
        ft_model = load_model(args.model_path, args)
        results["finetuned_model"] = evaluate_model(ft_model, tokenizer, args, run_dir)
        if gen_samples:
            logger.info(f"[2/2] Generation evaluation ({len(gen_samples)} samples)...")
            results["finetuned_generation"], ft_outputs = \
                generation_eval(ft_model, tokenizer, gen_samples, args)
        else:
            ft_outputs = None
        del ft_model
        torch.cuda.empty_cache()

        # 提升幅度
        b, f_ = results["base_model"], results["finetuned_model"]
        results["improvement"] = {
            "eval_loss_change": f_.get("eval_loss", 0) - b.get("eval_loss", 0),
            "eval_mean_token_accuracy_change":
                f_.get("eval_mean_token_accuracy", 0) - b.get("eval_mean_token_accuracy", 0),
        }

        # 生成对照样本 (人工抽查用)
        if gen_samples:
            compare = []
            for m, bo, fo in zip(gen_samples, base_outputs, ft_outputs):
                compare.append({
                    "context": m[:-1],
                    "reference": m[-1]['content'],
                    "base_output": bo,
                    "finetuned_output": fo,
                })
            with open(run_dir / "generations_compare.json", 'w', encoding='utf-8') as fp:
                json.dump(compare, fp, ensure_ascii=False, indent=2)
            logger.info(f"Generation samples saved to {run_dir / 'generations_compare.json'}")

        # 打印对比表: loss 指标 + 生成指标
        logger.info("=" * 72)
        logger.info("微调前后模型对比 (测试集)")
        logger.info("=" * 72)
        rows = [("eval_loss", results["base_model"], results["finetuned_model"],
                 "↓好 | 预测损失, 越低预测越准"),
                ("eval_mean_token_accuracy", results["base_model"], results["finetuned_model"],
                 "↑好 | 下一词预测准确率"),
                ("eval_entropy", results["base_model"], results["finetuned_model"],
                 "输出不确定性, 越低越笃定")]
        if gen_samples:
            rows += [("exact_match", results["base_generation"], results["finetuned_generation"],
                      "↑好 | 生成与标准答案完全一致的比例"),
                     ("bleu", results["base_generation"], results["finetuned_generation"],
                      "↑好 | n-gram重叠度, 精确率导向"),
                     ("rouge1", results["base_generation"], results["finetuned_generation"],
                      "↑好 | 词级重叠F1, 覆盖度导向"),
                     ("rougeL", results["base_generation"], results["finetuned_generation"],
                      "↑好 | 最长公共子序列F1, 句子级相似"),
                     ("ttft_ms", results["base_generation"], results["finetuned_generation"],
                      "↓好 | 首 token 延迟(毫秒), 越低响应越快"),
                     ("prefill_tokens_per_sec", results["base_generation"], results["finetuned_generation"],
                      "↑好 | 输入(prompt)处理速度, token/秒"),
                     ("decode_tokens_per_sec", results["base_generation"], results["finetuned_generation"],
                      "↑好 | 输出生成速度, token/秒")]
        logger.info(f"  {'指标':<30s}{'微调前':>12s}{'微调后':>12s}{'变化':>12s}  说明")
        for key, bd, fd, note in rows:
            if key in bd and key in fd:
                delta = fd[key] - bd[key]
                logger.info(f"  {key:<30s}{bd[key]:>12.4f}{fd[key]:>12.4f}"
                            f"{delta:>+12.4f}  {note}")
        logger.info("=" * 72)
    else:
        # ================= 单模型评估模式 =================
        model = load_model(args.model_path, args)
        results["model"] = evaluate_model(model, tokenizer, args, run_dir)
        if gen_samples:
            logger.info(f"Generation evaluation ({len(gen_samples)} samples)...")
            results["generation"], outputs = \
                generation_eval(model, tokenizer, gen_samples, args)
            gen = [{"context": m[:-1], "reference": m[-1]['content'], "output": o}
                   for m, o in zip(gen_samples, outputs)]
            with open(run_dir / "generations.json", 'w', encoding='utf-8') as fp:
                json.dump(gen, fp, ensure_ascii=False, indent=2)
        del model
        torch.cuda.empty_cache()
        logger.info(f"Test metrics: {json.dumps(results, default=str)[:300]}")

    # 保存全部结果
    test_path = run_dir / "test_results.json"
    with open(test_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Test results saved to {test_path}")

    return results


def main():
    """主函数"""
    args = parse_args()
    test(args)
    logger.info("Test done!")


if __name__ == "__main__":
    main()