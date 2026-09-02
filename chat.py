#!/usr/bin/env python3
"""
命令行对话测试脚本
加载底座模型或微调后的模型, 在终端交互式对话

用法:
  python chat.py                                       # 交互选择: 微调模型 / 原生模型
  python chat.py --model_path ./models/Qwen2.5-0.5B    # 跳过选择直接指定
  python chat.py --system "自定义系统提示"              # 换角色

对话中命令:
  /reset  清空对话历史    /exit  退出
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

DEFAULT_SYSTEM = "你是一个专业的电商客服，负责解答顾客关于商品、下单、快递、发货、退换货、优惠等问题。"


def run_sort_key(p):
    """按路径中的时间戳排序 (YYYYMMDD_HHMMSS), 兼容 run 目录或其子路径(如 final)"""
    s = str(p).replace("\\", "/")
    if "_train_" in s:
        return s.split("_train_")[-1].split("/")[0]
    return s


def find_latest_model(output_root: str = "./output"):
    """找最新微调结果 (按时间戳)"""
    candidates = sorted(Path(output_root).glob("*_train_*/final"), key=run_sort_key)
    return str(candidates[-1]) if candidates else None


def list_models(output_root: str = "./output", base_dir: str = "./models"):
    """扫描可选模型: [(显示名, 路径), ...]  微调模型在前(按时间新→旧), 原生模型在后"""
    models = []
    # 微调结果: 各训练 run 的 final + 各 epoch checkpoint (按时间新→旧)
    if Path(output_root).exists():
        for run in sorted(Path(output_root).glob("*_train_*"), key=run_sort_key, reverse=True):
            final = run / "final"
            if (final / "adapter_config.json").exists():
                models.append((f"微调模型  {run.name}/final", str(final)))
            for ckpt in sorted(run.glob("checkpoint-*"), key=lambda c: c.name, reverse=True):
                if (ckpt / "adapter_config.json").exists():
                    models.append((f"微调模型  {run.name}/{ckpt.name}", str(ckpt)))
    # 原生底座模型
    if Path(base_dir).exists():
        for d in sorted(Path(base_dir).iterdir(), reverse=True):
            if d.is_dir() and (d / "config.json").exists():
                models.append((f"原生模型  {d.name}", str(d)))
    return models


def choose_model(models):
    """交互选择模型, 返回 (显示名, 路径)"""
    print("可用模型:")
    for i, (name, _) in enumerate(models, 1):
        print(f"  {i}. {name}")
    print()
    while True:
        try:
            choice = input(f"选择模型编号 (回车=1): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not choice:
            return models[0]
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print(f"无效输入, 请输入 1~{len(models)} 的编号")


def load_model(model_path: str, fp16=True, bf16=False):
    """加载模型 (自动识别 LoRA adapter)"""
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype, "device_map": "auto"}
    if (Path(model_path) / "adapter_config.json").exists():
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="命令行对话测试")
    parser.add_argument("--model_path", type=str, default=None,
                       help="模型路径 (默认: 最新微调结果, 无则用底座模型)")
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM,
                       help="系统提示")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                       help="单次最大生成 token 数 (ECD 客服回复普遍短小)")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="采样温度 (0 = greedy)")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.15,
                       help="重复惩罚, 抑制复读 (1.0 = 关闭)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4,
                       help="禁止 n-gram 连续重复 (0 = 关闭)")
    parser.add_argument("--history_turns", type=int, default=5,
                       help="保留最近 N 轮历史")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--no_perf", action="store_true",
                       help="不显示生成速度")
    return parser.parse_args()


def main():
    args = parse_args()

    # 定位模型: --model_path 指定则直接用, 否则交互选择
    if args.model_path is None:
        models = list_models()
        if not models:
            raise SystemExit("未找到任何模型: 请先下载底座模型或完成一次训练")
        if len(models) == 1:
            args.model_path = models[0][1]
            print(f"模型: {models[0][0]}")
        else:
            picked = choose_model(models)
            if picked is None:
                print("再见!")
                return
            args.model_path = picked[1]
            print(f"已选择: {picked[0]}")
    else:
        print(f"模型: {args.model_path}")
    print(f"提示: /reset 清空历史 | /exit 退出")
    print("-" * 56)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model_path, args.fp16, args.bf16)

    history = []
    while True:
        try:
            user = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in ("/exit", "/quit", "退出"):
            break
        if user in ("/reset", "/new"):
            history = []
            print("(已清空对话历史)")
            continue

        # system + 最近 N 轮历史
        history.append({"role": "user", "content": user})
        msgs = ([{"role": "system", "content": args.system}]
                + history[-(args.history_turns * 2):])

        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_in = inputs['input_ids'].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size or None,
                pad_token_id=tokenizer.pad_token_id,
            )
        dt = time.perf_counter() - t0

        n_new = out.shape[1] - n_in
        reply = tokenizer.decode(
            out[0][n_in:], skip_special_tokens=True
        ).strip()
        history.append({"role": "assistant", "content": reply})

        perf = (f"  [{n_new} tokens | {dt:.1f}s | {n_new / dt:.1f} tok/s]"
                if not args.no_perf else "")
        print(f"客服:{perf}\n{reply}")

    print("再见!")


if __name__ == "__main__":
    main()