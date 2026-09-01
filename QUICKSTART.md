# 快速开始指南

## 环境要求

- Python 3.8+
- CUDA 11.7+ (可选，用于 GPU 训练)
- 16GB+ RAM
- 4GB+ GPU 显存 (LoRA) 或 8GB+ (QLoRA)

## 1. 安装依赖

```bash
cd qwen2.5-finetune
pip install -r requirements.txt
```

## 2. 下载模型

```bash
# 从 HuggingFace
python scripts/download_model.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --output_dir ./models/Qwen2.5-0.5B

# 或从 ModelScope（国内）
python scripts/download_model.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --output_dir ./models/Qwen2.5-0.5B \
    --source modelscope
```

## 3. 准备数据

```bash
# 预处理数据
python scripts/preprocess_data.py \
    --input ./data/raw_data.json \
    --output ./data/train.jsonl \
    --format chatml \
    --remove_duplicates \
    --split
```

## 4. 开始训练

```bash
# LoRA 训练
python main.py \
    --model_path ./models/Qwen2.5-0.5B \
    --data_path ./data/train.jsonl \
    --eval_data_path ./data/eval.jsonl \
    --output_dir ./output \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --use_wandb
```

## 5. 评估模型

```python
import asyncio
from evaluation.core import EvalEngine
from evaluation.datasets import BenchmarkRunner

engine = EvalEngine("evaluation/config/eval_config.yaml")
runner = BenchmarkRunner(engine)

results = asyncio.run(runner.run_all_benchmarks(
    model_path="./output/final",
    benchmarks=["ceval", "gsm8k"]
))
```

## 常见问题

### Q: 如何使用自己的数据？

A: 准备 JSONL 格式的数据：
```json
{"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
```

### Q: 显存不足怎么办？

A: 使用 QLoRA：
```bash
python main.py --use_qlora --batch_size 16
```

### Q: 如何查看训练日志？

A: 使用 TensorBoard 或 WandB：
```bash
tensorboard --logdir ./output
```
