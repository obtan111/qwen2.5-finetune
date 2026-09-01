#!/usr/bin/env python3
"""
Qwen 2.5 企业级微调脚本
支持 LoRA/QLoRA 训练、评估和报告生成
"""

import os
import sys
import json
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
    parser.add_argument("--model_path", type=str, default="./Qwen2.5-0.5B",
                       help="模型路径或 HuggingFace 模型 ID")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="输出目录")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                       help="训练数据路径 (jsonl 格式)")
    parser.add_argument("--eval_data_path", type=str, default=None,
                       help="评估数据路径")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="最大序列长度")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="学习率")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                       help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="梯度累积步数")
    
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
    parser.add_argument("--use_qlora", action="store_true",
                       help="是否使用 QLoRA (4-bit 量化)")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="是否以 4-bit 加载模型")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="是否以 8-bit 加载模型")
    
    # 监控参数
    parser.add_argument("--use_wandb", action="store_true",
                       help="是否使用 WandB")
    parser.add_argument("--wandb_project", type=str, default="qwen-finetune",
                       help="WandB 项目名")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="日志间隔步数")
    parser.add_argument("--eval_steps", type=int, default=50,
                       help="评估间隔步数")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="保存间隔步数")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--fp16", action="store_true",
                       help="是否使用 FP16")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="是否使用 BF16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="是否使用梯度检查点")
    
    return parser.parse_args()


def setup_wandb(args):
    """设置 WandB"""
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f"qwen2.5-0.5b-lora-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
            logger.info("WandB initialized")
        except ImportError:
            logger.warning("wandb not installed, skipping")


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
        texts = []
        for messages in examples.get('messages', []):
            if messages:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)
            else:
                # 尝试使用 prompt/reference 格式
                prompt = examples.get('prompt', [''])[0]
                reference = examples.get('reference', [''])[0]
                if prompt:
                    text = f"User: {prompt}\nAssistant: {reference}"
                    texts.append(text)
        
        return {"text": texts}
    
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
            bnb_4bit_compute_dtype=torch.bfloat16,
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


def setup_training_args(args):
    """设置训练参数"""
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.eval_data_path else "no",
        eval_steps=args.eval_steps if args.eval_data_path else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="wandb" if args.use_wandb else "tensorboard",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        optim="paged_adamw_8bit" if (args.use_qlora or args.load_in_4bit) else "adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        seed=args.seed,
    )
    
    return training_args


def train(args):
    """训练主函数"""
    logger.info("Starting training...")
    logger.info(f"Arguments: {vars(args)}")
    
    # 设置 WandB
    setup_wandb(args)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # 训练参数
    training_args = setup_training_args(args)
    
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
    
    return model, tokenizer


def evaluate(args):
    """评估"""
    logger.info("Starting evaluation...")
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    from evaluation.core import EvalEngine
    from evaluation.datasets import BenchmarkRunner
    
    # 加载评估配置
    config_path = Path(__file__).parent / "evaluation" / "config" / "eval_config.yaml"
    if config_path.exists():
        eval_engine = EvalEngine(str(config_path))
        benchmark_runner = BenchmarkRunner(eval_engine)
        
        # 运行评估
        import asyncio
        results = asyncio.run(benchmark_runner.run_all_benchmarks(
            model_path=f"{args.output_dir}/final",
            benchmarks=["ceval", "gsm8k"]
        ))
        
        # 保存结果
        output_path = Path(args.output_dir) / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")
    else:
        logger.warning("Evaluation config not found, skipping evaluation")


def main():
    """主函数"""
    args = parse_args()
    
    # 训练
    model, tokenizer = train(args)
    
    # 评估（可选）
    if not args.use_wandb:  # 如果没有用 wandb，执行评估
        try:
            evaluate(args)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    
    logger.info("All done!")


if __name__ == "__main__":
    main()
