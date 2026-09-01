# Qwen 2.5 企业级微调框架

> 面向生产环境的 Qwen 2.5 模型微调、评估和部署框架

## 特性

- 🚀 **多种训练方式**: 支持 LoRA、QLoRA、全量微调
- 📊 **全面的评估体系**: 自动化指标、LLM-as-Judge、安全性评估
- 📈 **标准 Benchmark**: MMLU、GSM8K、CEval、HumanEval 等
- 📝 **数据预处理**: 支持多种格式转换、清洗、去重
- 📊 **可视化报告**: 训练曲线、指标对比、安全评估报告
- 🔧 **易于扩展**: 模块化设计，支持自定义指标和评估器

## 项目结构

```
qwen2.5-finetune/
├── evaluation/                   # 评估框架
│   ├── config/                   # 配置文件
│   │   ├── eval_config.yaml      # 评估配置
│   │   └── model_config.yaml     # 模型配置
│   ├── core/                     # 核心模块
│   │   ├── eval_engine.py        # 评估引擎
│   │   ├── model_manager.py      # 模型管理
│   │   ├── cache_manager.py      # 缓存管理
│   │   └── resource_monitor.py   # 资源监控
│   ├── metrics/                  # 指标计算
│   │   ├── base_metrics.py       # 基础指标
│   │   ├── generation_metrics.py # 生成指标
│   │   ├── task_metrics.py       # 任务指标
│   │   └── statistical_metrics.py # 统计指标
│   ├── datasets/                 # 数据集管理
│   │   ├── dataset_loader.py     # 数据加载
│   │   ├── dataset_validator.py  # 数据验证
│   │   └── benchmark_runner.py   # Benchmark运行
│   ├── evaluators/               # 评估器
│   │   ├── llm_as_judge.py       # LLM评估器
│   │   ├── safety_eval.py        # 安全评估
│   │   └── human_eval_bridge.py  # 人工评估
│   └── reporting/                # 报告生成
│       ├── report_generator.py   # 报告生成
│       ├── visualization.py      # 可视化
│       └── export_utils.py       # 导出工具
├── scripts/                      # 工具脚本
│   ├── download_model.py         # 模型下载
│   └── preprocess_data.py        # 数据预处理
├── main.py                       # 训练主入口
├── requirements.txt              # 依赖
└── README.md                     # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

```bash
# 从 HuggingFace 下载
python scripts/download_model.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --output_dir ./models/Qwen2.5-0.5B

# 从 ModelScope 下载（国内推荐）
python scripts/download_model.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --output_dir ./models/Qwen2.5-0.5B \
    --source modelscope
```

### 3. 准备数据

#### 数据格式

支持多种格式，推荐使用 ChatML 格式：

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "什么是机器学习？"},
      {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
    ]
  }
]
```

#### 数据预处理

```bash
# 清洗和格式化数据
python scripts/preprocess_data.py \
    --input ./data/raw_data.json \
    --output ./data/processed_data.jsonl \
    --format chatml \
    --remove_duplicates \
    --min_length 10 \
    --max_length 4096 \
    --split
```

### 4. 训练模型

#### LoRA 训练（推荐）

```bash
python main.py \
    --model_path ./models/Qwen2.5-0.5B \
    --data_path ./data/train.jsonl \
    --eval_data_path ./data/eval.jsonl \
    --output_dir ./output \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --use_wandb
```

#### QLoRA 训练（显存有限）

```bash
python main.py \
    --model_path ./models/Qwen2.5-0.5B \
    --data_path ./data/train.jsonl \
    --output_dir ./output \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --use_qlora \
    --use_wandb
```

### 5. 评估模型

```python
import asyncio
from evaluation.core import EvalEngine
from evaluation.datasets import BenchmarkRunner

# 加载评估引擎
engine = EvalEngine("evaluation/config/eval_config.yaml")
runner = BenchmarkRunner(engine)

# 运行评估
results = asyncio.run(runner.run_all_benchmarks(
    model_path="./output/final",
    benchmarks=["ceval", "gsm8k"]
))

# 保存结果
import json
with open("results/evaluation.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

## 训练参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | - | 模型路径或 HF ID |
| `--data_path` | - | 训练数据路径 |
| `--output_dir` | ./output | 输出目录 |
| `--num_epochs` | 3 | 训练轮数 |
| `--batch_size` | 8 | 批大小 |
| `--learning_rate` | 2e-4 | 学习率 |

### LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |

### 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_qlora` | False | 启用 QLoRA |
| `--load_in_4bit` | False | 4-bit 量化 |
| `--load_in_8bit` | False | 8-bit 量化 |

## 评估指标

### 自动化指标

- **Perplexity**: 困惑度
- **Accuracy**: 准确率
- **F1 Score**: F1 分数
- **BLEU**: 机器翻译评估
- **ROUGE**: 文本摘要评估
- **METEOR**: 生成评估

### 模型评估

- **LLM-as-Judge**: 使用 GPT-4 等模型评估
- **Relevance**: 相关性评分
- **Coherence**: 连贯性评分
- **Factuality**: 事实性评分
- **Helpfulness**: 有用性评分

### 安全性评估

- **Toxicity Score**: 毒性检测
- **Bias Score**: 偏见检测
- **Hallucination Rate**: 幻觉率
- **Refusal Rate**: 拒绝率

### 性能指标

- **Latency**: 响应延迟
- **Throughput**: 吞吐量
- **Memory Usage**: 内存使用

## Benchmark

支持的标准 Benchmark：

| Benchmark | 说明 | 指标 |
|-----------|------|------|
| MMLU | 多任务语言理解 | Accuracy |
| GSM8K | 数学推理 | Accuracy |
| CEval | 中文评估 | Accuracy |
| CMMLU | 中文多任务 | Accuracy |
| HellaSwag | 常识推理 | Accuracy |
| HumanEval | 代码生成 | pass@k |

## 配置说明

### 评估配置 (eval_config.yaml)

```yaml
evaluation:
  model:
    path: "./Qwen2.5-0.5B"
    torch_dtype: "bfloat16"
  
  datasets:
    - name: "custom_eval"
      path: "./data/eval.jsonl"
      max_samples: 1000
  
  generation:
    max_new_tokens: 1024
    temperature: 0.7
    top_p: 0.9
  
  metrics:
    automated:
      - "accuracy"
      - "f1_score"
      - "bleu"
      - "rouge_l"
```

## 监控和日志

### WandB 集成

```bash
# 启用 WandB
python main.py --use_wandb --wandb_project my_project
```

### TensorBoard

```bash
# 查看训练日志
tensorboard --logdir ./output
```

## 常见问题

### Q: 显存不足怎么办？

A: 使用 QLoRA（4-bit 量化）：
```bash
python main.py --use_qlora --batch_size 16
```

### Q: 如何评估模型效果？

A: 运行标准 Benchmark：
```python
from evaluation.datasets import BenchmarkRunner
runner = BenchmarkRunner(engine)
results = await runner.run_benchmark("ceval", model_path)
```

### Q: 如何添加自定义评估指标？

A: 在 `evaluation/metrics/` 中添加新的指标类，并在 `MetricsCalculator` 中注册。

## License

MIT License
