#!/usr/bin/env python3
"""
Qwen 2.5 微调训练脚本 (训练 + 验证)
支持 LoRA/QLoRA 训练, 训练后在验证集上评估, 结果全部保存到本地
测试集评估请使用 test.py
"""

import os
import sys
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen 2.5 Fine-tuning")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="./models/Qwen2.5-0.5B",
                       help="模型路径或 HuggingFace 模型 ID")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="输出目录")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, default="./data/ecd/train_10k.jsonl",
                       help="训练数据路径 (默认 1 万条抽样, 全量用 ./data/ecd/train.jsonl)")
    parser.add_argument("--eval_data_path", type=str, default="./data/ecd/eval.jsonl",
                       help="验证数据路径 (训练中/后评估)")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                       help="最大序列长度 (ECD 最长样本约 1000 token, 1024 足够且省显存)")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数 (GTX 1660 SUPER 上 1 万条约 4-6 小时/轮, 先 1 轮看效果)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批大小 (ECD 短文本, 16 可提升 GPU 利用率)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="学习率")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                       help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数 (batch 16 时无需累积)")
    
    # LoRA 参数
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="是否使用 LoRA")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # 量化参数
    parser.add_argument("--use_qlora", action="store_true", default=False,
                       help="是否使用 QLoRA (4-bit 量化)")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                       help="是否以 4-bit 加载模型")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                       help="是否以 8-bit 加载模型")
    
    # 监控参数
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="日志间隔步数")
    parser.add_argument("--eval_steps", type=int, default=250,
                       help="验证间隔步数 (1 万条 batch16 约 625 步/轮)")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="保存间隔步数 (仅 save_strategy=steps 时生效)")
    parser.add_argument("--save_strategy", type=str, default="epoch",
                       choices=["epoch", "steps", "no"],
                       help="检查点保存策略 (epoch=每轮保存, steps=按步保存, no=不保存)")
    parser.add_argument("--save_total_limit", type=int, default=None,
                       help="最多保留检查点数 (默认不限制, LoRA 检查点很小)")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="是否使用 FP16")
    parser.add_argument("--bf16", action="store_true", default=False,
                       help="是否使用 BF16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="是否使用梯度检查点")
    
    return parser.parse_args()


def get_dataset_name(data_path: str) -> str:
    """从数据路径提取数据集名: ./data/ecd/train.jsonl -> ecd"""
    p = Path(data_path)
    name = p.parent.name
    if name in ("data", ".", "..", ""):
        name = p.stem
    return name


def load_and_preprocess_data(data_path: str, tokenizer, max_seq_length: int):
    """加载和预处理数据"""
    logger.info(f"Loading data from {data_path}")
    
    # 加载数据
    if data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path)
    elif data_path.endswith('.json'):
        dataset = load_dataset('json', data_files=data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # 预处理
    def preprocess(examples):
        # 输出 prompt/completion 两列, 配合 completion_only_loss:
        # loss 只算客服回复部分, 不学习顾客的话 (避免模型变成"对话流生成器")
        prompts, completions = [], []
        if 'messages' in examples and examples['messages']:
            # ChatML 格式: prompt=对话历史, completion=最后一条 assistant 回复
            for messages in examples['messages']:
                if messages and messages[-1]['role'] == 'assistant':
                    prompt = tokenizer.apply_chat_template(
                        messages[:-1], tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(prompt)
                    completions.append(messages[-1]['content'])
                else:
                    prompts.append("")
                    completions.append("")
        else:
            # prompt/reference 格式
            qs = examples.get('prompt') or examples.get('instruction') or []
            refs = (examples.get('reference') or examples.get('answer')
                    or examples.get('output') or [])
            for q, r in zip(qs, refs):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q or ""}],
                    tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
                completions.append(r or "")
        
        return {"prompt": prompts, "completion": completions}
    
    processed_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    logger.info(f"Processed {len(processed_dataset['train'])} samples")
    return processed_dataset


def setup_model_and_tokenizer(args):
    """设置模型和 tokenizer"""
    logger.info(f"Loading model from {args.model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 量化配置
    bnb_config = None
    if args.use_qlora or args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        "device_map": "auto",
    }
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
    )
    
    # 配置 LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def setup_training_args(args, num_train_samples=None):
    """设置训练参数"""
    # transformers 5.x 已移除 warmup_ratio, 需换算为 warmup_steps
    warmup_steps = 0
    if num_train_samples:
        steps_per_epoch = max(1, math.ceil(
            num_train_samples / (args.batch_size * args.gradient_accumulation_steps)
        ))
        total_steps = steps_per_epoch * args.num_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.eval_data_path else "no",
        eval_steps=args.eval_steps if args.eval_data_path else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="tensorboard",
        max_length=args.max_seq_length,
        completion_only_loss=True,   # 只对回复部分算 loss, 不学习顾客的话
        optim="paged_adamw_8bit" if (args.use_qlora or args.load_in_4bit) else "adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )
    
    return training_args


def train(args):
    """训练主函数"""
    logger.info("Starting training...")
    logger.info(f"Arguments: {vars(args)}")
    
    # 创建本次运行的输出目录: {数据集名}_train_{时间戳} (每次训练独立, 不覆盖历史结果)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = get_dataset_name(args.data_path)
    run_dir = Path(args.output_dir) / f"{dataset_name}_train_{timestamp}"
    args.output_dir = str(run_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Run directory: {args.output_dir}")
    
    # 加载模型
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # 加载数据
    dataset = load_and_preprocess_data(
        args.data_path,
        tokenizer,
        args.max_seq_length
    )
    
    # 加载评估数据
    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = load_and_preprocess_data(
            args.eval_data_path,
            tokenizer,
            args.max_seq_length
        )['train']
    
    # 训练参数 (传入样本数用于换算 warmup_steps)
    training_args = setup_training_args(args, num_train_samples=len(dataset['train']))
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # 训练
    logger.info("Starting training...")
    trainer.train()
    
    # 保存
    final_output_dir = f"{args.output_dir}/final"
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    logger.info(f"Training completed! Model saved to {final_output_dir}")
    
    # 保存训练配置
    config_path = f"{args.output_dir}/training_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    logger.info(f"Training config saved to {config_path}")
    
    # 保存训练历史 (loss 曲线、学习率等) 到本地 JSON
    history_path = f"{args.output_dir}/train_history.json"
    with open(history_path, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)
    logger.info(f"Training history saved to {history_path}")
    
    # 本地评估: 验证集
    if eval_dataset is not None:
        logger.info("Evaluating on eval dataset...")
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        eval_path = f"{args.output_dir}/eval_results.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        logger.info(f"Eval results saved to {eval_path}")
    
    return model, tokenizer


def main():
    """主函数"""
    args = parse_args()
    
    # 训练 + 验证 (结果保存到本地, 测试集评估请用 test.py)
    train(args)
    
    logger.info("All done!")


if __name__ == "__main__":
    main()
