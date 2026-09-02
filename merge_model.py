#!/usr/bin/env python3
"""
LoRA 合并脚本: 把微调的 adapter 权重合并进底座模型, 导出独立完整模型
合并后 (1) 推理速度恢复原生水平 (2) 可脱离底座单独部署 (3) 质量无损 (数学等价)

用法:
  python merge_model.py                                    # 合并最新微调结果
  python merge_model.py --model_path output/ecd_train_xxx/final
  python merge_model.py --output_dir models/my-merged-model
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM


def run_sort_key(p):
    """按路径中的时间戳排序 (YYYYMMDD_HHMMSS), 兼容 run 目录或其子路径(如 final)"""
    s = str(p).replace("\\", "/")
    if "_train_" in s:
        return s.split("_train_")[-1].split("/")[0]
    return s


def find_latest_model(output_root: str = "./output"):
    candidates = sorted(Path(output_root).glob("*_train_*/final"), key=run_sort_key)
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError(f"No trained model found under {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Merge Tool")
    parser.add_argument("--model_path", type=str, default=None,
                       help="LoRA adapter 路径 (默认: 最新训练结果 final)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="合并模型保存目录 (默认: models/{run名}-merged)")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="合并后保存精度 (默认 float16)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_path is None:
        args.model_path = find_latest_model()
    if (Path(args.model_path) / "adapter_config.json").exists() is False:
        raise SystemExit(f"Not a LoRA adapter: {args.model_path}")

    if args.output_dir is None:
        run_name = Path(args.model_path).parent.name  # e.g. ecd_train_20260902_005110
        args.output_dir = str(Path("models") / f"{run_name}-merged")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Adapter   : {args.model_path}")
    print(f"Output    : {args.output_dir}")
    print(f"精度      : {args.dtype}")

    # 在 CPU 上加载并合并 (不占显存, 小模型几秒钟)
    print("Loading adapter (CPU)...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cpu",
    )

    print("Merging LoRA weights into base model...")
    t0 = time.perf_counter()
    merged = model.merge_and_unload()   # W' = W + B@A, 之后结构与原生完全一致
    print(f"Merged in {time.perf_counter() - t0:.1f}s")

    print("Saving merged model...")
    merged.save_pretrained(args.output_dir)

    # tokenizer 一并保存 (从 adapter 目录取, 里面已含 tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.save_pretrained(args.output_dir)

    size_mb = sum(f.stat().st_size for f in Path(args.output_dir).rglob("*")
                  if f.is_file()) / 1e6
    print(f"\nDone! 合并模型已保存: {args.output_dir} ({size_mb:.0f} MB)")
    print("现在可在 chat.py 的模型列表中选择它, 推理速度恢复原生水平")


if __name__ == "__main__":
    main()