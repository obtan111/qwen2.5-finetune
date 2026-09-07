#!/usr/bin/env python3
"""
Qwen 2.5 微调测试脚本
=====================
这个脚本负责"考试"：训练完之后，检验微调到底有没有用。

它做两件事：
1. Loss 指标评估 —— 用"预测下一个词"的方式衡量模型对客服话术的掌握程度
2. 生成质量评估 —— 让模型真正开口回答，再和标准答案比对 (Exact Match / BLEU / ROUGE)

微调前 (基座模型) 和微调后 (LoRA 模型) 都会被评估一遍，结果对比着看。

结果保存到 output/{数据集名}_test_{时间戳}/ 目录（时间戳保证每次运行互不覆盖）。

用法:
  python test.py                                        # 自动找最新训练结果 + 对比基座
  python test.py --test_data_path ./data/ecd/test.jsonl # 指定测试数据
  python test.py --max_gen_samples 200                  # 生成评估样本数(默认100)
  python test.py --skip_base                            # 只测微调后，不测基座
"""

import argparse      # 解析命令行参数（--xxx 这种）
import json          # 读写 JSON 文件（结果都存成 JSON）
import logging       # 打印带时间戳的日志（控制台那些 INFO 行）
import os            # 文件/目录操作
import time          # 计时：测模型响应速度用
from datetime import datetime   # 生成时间戳，用于输出目录命名
from pathlib import Path        # 跨平台路径处理

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer   # 加载普通模型
from peft import AutoPeftModelForCausalLM                     # 加载带 LoRA 适配器的模型
from trl import SFTTrainer, SFTConfig                         # 复用训练器的"评估"能力

# 从 train.py 复用两个函数：数据预处理 + 从路径提取数据集名
from train import load_and_preprocess_data, get_dataset_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ====================================================================
# 基础工具
# ====================================================================

def run_sort_key(p):
    """按路径中的时间戳排序 (YYYYMMDD_HHMMSS)

    训练输出目录长这样：output/ecd_train_20260906_155252/
    我们要按"20260906_155252"这个时间戳找"最新"的训练结果，
    所以从路径里把 _train_ 后面的时间部分切出来当作排序键。

    兼容两种输入：
      - output/ecd_train_xxx/final      （直接给 final 路径）
      - output/ecd_train_xxx            （给 run 目录本身）
    """
    s = str(p).replace("\\", "/")          # Windows 路径分隔符统一成 /
    if "_train_" in s:
        # 取 "_train_" 后面、第一个 "/" 前面的部分 = 时间戳
        return s.split("_train_")[-1].split("/")[0]
    return s                               # 没有时间戳就直接用原路径


def find_latest_model(output_root: str) -> str:
    """在 output 根目录下找最新训练 run 的 final 模型（按时间戳）

    例：output 下同时有
      output/ecd_train_20260901_100000/final
      output/ecd_train_20260906_155252/final
    会返回时间戳最新的那个（20260906...）。
    这样你不需要手动指定模型路径，每次跑 test.py 都测最新一次训练。
    """
    candidates = sorted(Path(output_root).glob("*_train_*/final"), key=run_sort_key)
    if candidates:
        return str(candidates[-1])          # 排序后最后一个 = 时间戳最新
    raise FileNotFoundError(
        f"No trained model found under {output_root}, run train.py first"
    )


def resolve_base_path(base_path: str):
    """解析基座模型路径

    训练时 LoRA adapter 的配置文件里记了基座模型路径（base_model_name_or_path），
    但记的可能是相对路径（如 ./models/Qwen2.5-0.5B）。
    运行 test.py 时工作目录可能变了，所以要"兜底解析"：
    1. 先按原样判断路径存不存在
    2. 不存在就把项目根目录拼上去再试一次
    """
    p = Path(base_path)
    if p.exists():
        return str(p)
    project_root = Path(__file__).resolve().parent   # test.py 所在目录 = 项目根
    p2 = project_root / base_path
    if p2.exists():
        return str(p2)
    return None     # 都找不到就返回 None，主流程会跳过对比模式


def load_model(model_path: str, args):
    """加载模型（自动识别 LoRA adapter 或完整模型）

    关键区别：
    - 如果目录里有 adapter_config.json → 这是 LoRA 微调产物
      → 用 AutoPeftModelForCausalLM 加载（基座 + LoRA 补丁）
    - 否则 → 普通完整模型 → 用 AutoModelForCausalLM 加载

    注意：加载 LoRA 模型时，模型前向计算会多跑一小段低秩矩阵运算
    （LoRA 的本质 = 在原权重旁并联一个小矩阵），
    这就是测试表里"微调后每 token 稍慢"的原因。部署时用 merge_model.py
    把 LoRA 合并回主模型就没有这个开销了。
    """
    # 精度选择：默认 fp16（半精度），显存省一半、速度更快
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model_kwargs = {
        "trust_remote_code": True,      # 允许加载社区模型的自定义代码
        "torch_dtype": dtype,           # 加载时的权重精度
        "device_map": "auto",           # 自动把模型放到 GPU/CPU
    }
    if (Path(model_path) / "adapter_config.json").exists():
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    return model


def load_raw_samples(data_path: str, max_samples: int):
    """读取原始 messages 样本（用于生成评估），统一为 messages 格式

    为什么不用上面的 load_and_preprocess_data？
    那个函数会把数据"分词打标签"，只适合算 loss；
    这里我们要拿到**原始对话文本**让模型真的开口生成，
    所以自己读文件，只做格式统一：
    - jsonl 格式：每行一个 JSON
    - json 格式：整个文件是一个列表
    最后统一成 [{role, content}...] 的对话格式。
    """
    p = Path(data_path)
    samples = []
    if p.suffix == '.jsonl':
        for line in open(p, encoding='utf-8'):      # 逐行读
            if line.strip():                         # 跳过空行
                samples.append(json.loads(line))
    else:
        samples = json.load(open(p, encoding='utf-8'))

    msgs = []
    for item in samples[:max_samples]:               # 只取前 max_samples 条
        if 'messages' in item:
            msgs.append(item['messages'])            # 已经是对话格式，直接用
        else:  # prompt/reference 格式 → 手动包成对话格式
            msgs.append([
                {"role": "user", "content": item.get('prompt', item.get('instruction', ''))},
                {"role": "assistant",
                 "content": item.get('reference', item.get('answer', item.get('output', '')))},
            ])
    return msgs


# ====================================================================
# Loss 指标评估（模型"会不会说"）
# ====================================================================

def evaluate_model(model, tokenizer, args, run_dir: Path):
    """构建评估 trainer 并返回 loss 类指标

    这里的思路：把测试数据喂给模型，让模型预测"客服下一句该说什么"，
    算预测和真实回复的差距（eval_loss）和猜对率（token accuracy）。

    小技巧：评估不需要真训练，但直接复用 SFTTrainer 的 evaluate()
    比自己写前向循环省事，所以临时构造一个只用于评估的 trainer。
    """
    # 复用 train.py 的预处理：把对话拆成 prompt + completion（只对客服回复算 loss）
    test_dataset = load_and_preprocess_data(
        args.test_data_path, tokenizer, args.max_seq_length
    )['train']

    # 评估专用的训练参数：不训练、不打日志、不存 checkpoint
    eval_args = SFTConfig(
        output_dir=str(run_dir),
        per_device_eval_batch_size=args.batch_size,   # 评估批大小（默认8）
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",                             # 不上报任何实验追踪平台
        max_length=args.max_seq_length,
        dataset_text_field="text",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=eval_args,
        train_dataset=test_dataset,   # 评估时传测试集当训练集，纯属占位
        eval_dataset=test_dataset,    # 真正评估的是这个
        processing_class=tokenizer,
    )
    return trainer.evaluate()         # 返回 {eval_loss, eval_entropy, ...}


# ====================================================================
# 生成质量评估（模型"会不会说好"）
# ====================================================================

def generate_responses(model, tokenizer, samples, args):
    """逐条生成回复，并统计性能指标

    对每条对话：
      1. 用上下文（顾客的话）做输入
      2. 模型生成回复
      3. 顺便测三个速度指标：
         - TTFT (time to first token)：首 token 延迟 = 用户感知的"响应速度"
         - prefill 吞吐：处理输入的速度
         - decode 吞吐：逐字生成输出的速度

    性能指标的测法（理解这段就理解了速度差异的由来）：
      - 先让模型只生成 1 个 token（max_new_tokens=1），测出"读完输入+吐第一个字"的时间 = TTFT
      - 再让它完整生成，总时间减去 TTFT = 纯生成时间，除以生成字数 = decode 速度
    """
    model.eval()                      # 切到评估模式（关闭 dropout 等训练行为）
    outputs, perf = [], []
    for i, messages in enumerate(samples):
        # 把对话历史（不含最后一句标准答案）套上对话模板变成 prompt
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_in = inputs['input_ids'].shape[1]       # 输入的长度（token 数）

        # 1) 单独测首 token 延迟：prefill（处理输入）+ 1 步 decode（吐第 1 个字）
        t0 = time.perf_counter()
        with torch.no_grad():                     # 不计算梯度，省显存加快速度
            model.generate(
                **inputs, max_new_tokens=1, do_sample=False,   # 只生成 1 个 token
                pad_token_id=tokenizer.pad_token_id,
            )
        ttft = time.perf_counter() - t0

        # 2) 完整生成回复（greedy 解码，保证每次结果可复现）
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,   # 最多生成多少 token（默认128）
                do_sample=False,                      # False=贪心，不抽样，结果稳定
                pad_token_id=tokenizer.pad_token_id,
            )
        total = time.perf_counter() - t0
        n_out = out.shape[1] - n_in                  # 实际生成了多少个新 token

        # 把生成的 token 解码回文字（去掉特殊符号）
        text = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        outputs.append(text.strip())
        perf.append({
            "input_tokens": n_in,
            "output_tokens": n_out,
            "ttft_ms": ttft * 1000,                              # 毫秒
            "prefill_tokens_per_sec": n_in / ttft if ttft > 0 else 0.0,   # 输入处理速度
            "decode_tokens_per_sec": max(n_out - 1, 1) / max(total - ttft, 1e-6),  # 输出生成速度
            "total_time_sec": total,
        })
        if (i + 1) % 20 == 0:                       # 每 20 条报一次进度
            logger.info(f"  generated {i + 1}/{len(samples)}")
    return outputs, perf


def compute_generation_metrics(references, hypotheses):
    """计算生成指标：exact_match / char-BLEU / ROUGE

    三个指标的含义：
    - exact_match：生成回复和标准答案【一字不差】的比例（最严格）
    - BLEU：n-gram 重叠度（看"关键词/短语"重合多少，偏精确率）
    - ROUGE：词级重叠 F1（看"该说的点"覆盖多少，偏召回率）

    中文处理的要点：中文没有空格分词，所以按【字符】切分后
    再计算（list("你好世界") = ['你','好','世','界']）。
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

    n = max(len(references), 1)                     # 防除零
    # exact_match：完全相等的比例
    exact = sum(1 for r, h in zip(references, hypotheses)
                if r.strip() == h.strip())
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    smooth = SmoothingFunction().method1            # BLEU 平滑（短句防零分）
    bleus, r1s, rl_s = [], [], []
    for r, h in zip(references, hypotheses):
        rc, hc = list(r.strip()), list(h.strip())   # 按字符切分
        try:
            bleus.append(sentence_bleu([rc], hc, smoothing_function=smooth))
        except Exception:
            bleus.append(0.0)                       # 极短句可能报错，兜底给 0
        s = scorer.score(' '.join(rc), ' '.join(hc))  # ROUGE 需要空格分隔输入
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
    """生成 + 质量指标 + 性能指标，返回 (metrics, outputs)

    把上面两个函数串起来：先生成，再算指标，再把速度指标
    从"每条一个"汇总成"平均一个"，方便对比表直接展示。
    """
    outputs, perf = generate_responses(model, tokenizer, samples, args)
    references = [m[-1]['content'] for m in samples]   # 标准答案 = 每条对话最后一句
    metrics = compute_generation_metrics(references, outputs)

    # 汇总性能指标（取平均，平铺进同一个 metrics 字典）
    if perf:
        n = len(perf)
        metrics["ttft_ms"] = sum(p["ttft_ms"] for p in perf) / n
        metrics["prefill_tokens_per_sec"] = sum(p["prefill_tokens_per_sec"] for p in perf) / n
        metrics["decode_tokens_per_sec"] = sum(p["decode_tokens_per_sec"] for p in perf) / n
        metrics["avg_input_tokens"] = sum(p["input_tokens"] for p in perf) / n
        metrics["avg_output_tokens"] = sum(p["output_tokens"] for p in perf) / n
    return metrics, outputs


# ====================================================================
# 参数
# ====================================================================

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


# ====================================================================
# 主流程
# ====================================================================

def test(args):
    """测试主函数：loss 指标 + 生成质量，微调前后对比"""
    logger.info("Starting testing...")

    # 没指定模型路径就自动找最新的训练结果
    if args.model_path is None:
        args.model_path = find_latest_model(args.output_dir)
    logger.info(f"Model to test: {args.model_path}")

    # 创建本次测试的输出目录（时间戳命名，不覆盖历史结果）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = get_dataset_name(args.test_data_path)
    run_dir = Path(args.output_dir) / f"{dataset_name}_test_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # 加载分词器（两个模型共用同一个 tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # 补 pad_token，否则批量生成会报错

    # 读取用于"生成评估"的原始样本（--max_gen_samples 0 则关闭生成评估）
    gen_samples = load_raw_samples(args.test_data_path, args.max_gen_samples) \
        if args.max_gen_samples > 0 else []

    # 判断要测的模型是不是 LoRA 产物（目录里有没有 adapter_config.json）
    is_adapter = (Path(args.model_path) / "adapter_config.json").exists()
    results = {}

    # 从 adapter 配置里找到基座模型路径（用于微调前对比）
    base_path = None
    if is_adapter and not args.skip_base:
        adapter_cfg = json.load(open(Path(args.model_path) / "adapter_config.json",
                                     encoding='utf-8'))
        base_path = resolve_base_path(adapter_cfg.get("base_model_name_or_path", ""))

    if base_path:
        # ================= 微调前后对比模式（默认走这里） =================
        # [1/2] 先测基座模型（微调前）：loss 指标 + 生成质量
        logger.info(f"[1/2] Evaluating BASE model: {base_path}")
        base_model = load_model(base_path, args)
        results["base_model"] = evaluate_model(base_model, tokenizer, args, run_dir)
        if gen_samples:
            logger.info(f"[1/2] Generation evaluation ({len(gen_samples)} samples)...")
            results["base_generation"], base_outputs = \
                generation_eval(base_model, tokenizer, gen_samples, args)
        else:
            base_outputs = None
        del base_model                      # 释放显存
        torch.cuda.empty_cache()            # 清空 GPU 缓存，给第二个模型腾地方

        # [2/2] 再测微调后模型：同样的两套评估
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

        # 计算提升幅度（loss 类的变化量，保存到结果 JSON 里）
        b, f_ = results["base_model"], results["finetuned_model"]
        results["improvement"] = {
            "eval_loss_change": f_.get("eval_loss", 0) - b.get("eval_loss", 0),
            "eval_mean_token_accuracy_change":
                f_.get("eval_mean_token_accuracy", 0) - b.get("eval_mean_token_accuracy", 0),
        }

        # 保存"问题 + 标准答案 + 微调前回复 + 微调后回复"的对照样本
        # → generations_compare.json，供人工抽查（比看数字更直观）
        if gen_samples:
            compare = []
            for m, bo, fo in zip(gen_samples, base_outputs, ft_outputs):
                compare.append({
                    "context": m[:-1],                 # 对话上下文（顾客的话）
                    "reference": m[-1]['content'],     # 标准答案（客服应该怎么回）
                    "base_output": bo,                 # 微调前模型的回复
                    "finetuned_output": fo,            # 微调后模型的回复
                })
            with open(run_dir / "generations_compare.json", 'w', encoding='utf-8') as fp:
                json.dump(compare, fp, ensure_ascii=False, indent=2)
            logger.info(f"Generation samples saved to {run_dir / 'generations_compare.json'}")

        # 打印对比表：三行 loss 类指标 + 五行生成/性能指标
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
                delta = fd[key] - bd[key]            # 变化量 = 微调后 - 微调前
                logger.info(f"  {key:<30s}{bd[key]:>12.4f}{fd[key]:>12.4f}"
                            f"{delta:>+12.4f}  {note}")
        logger.info("=" * 72)
    else:
        # ================= 单模型评估模式 =================
        # 两种情况走到这：测的不是 LoRA 产物，或 --skip_base
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

    # 保存全部结果（loss 指标 + 生成指标 + 提升幅度都在里面）
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
