#!/usr/bin/env python3
"""
命令行对话测试脚本
加载底座模型或微调后的模型, 在终端交互式对话

用法:
  python chat.py                                       # 用最新微调结果(无则用底座)
  python chat.py --model_path ./models/Qwen2.5-0.5B    # 指定用底座模型对比
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


def find_latest_model(output_root: str = "./output"):
    """找最新微调结果"""
    candidates = sorted(Path(output_root).glob("*_train_*/final"))
    return str(candidates[-1]) if candidates else None


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
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="单次最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="采样温度 (0 = greedy)")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--history_turns", type=int, default=5,
                       help="保留最近 N 轮历史")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--no_perf", action="store_true",
                       help="不显示生成速度")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_path is None:
        args.model_path = find_latest_model() or "./models/Qwen2.5-0.5B"
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