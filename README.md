# Qwen 2.5 电商客服微调项目

基于真实电商客服对话语料（ECD，淘宝）微调 Qwen2.5-0.5B，提供完整的**训练 → 验证 → 测试对比 → LLM 评审**流程，所有结果本地保存、按运行隔离、互不覆盖。



---

## 环境（已配置）

| 项 | 值 |
|---|---|
| conda 环境 | `qwen25`（Python 3.10，`conda activate qwen25`） |
| PyTorch | 2.5.1+cu121 |
| 关键依赖 | transformers 5.16.1 / trl 1.12.0 / peft 0.20.0 / bitsandbytes 0.50.2 |
| GPU | GTX 1660 SUPER（6GB，Turing 架构，**不支持 bf16，默认 fp16**） |
| 底座模型 | `models/Qwen2.5-0.5B`（4.94 亿参数，bf16 原始权重） |

---

## 项目结构

```
qwen2.5-finetune/
├── train.py                       # 训练脚本（训练+验证, 每个 epoch 保存检查点）
├── test.py                        # 测试脚本（微调前后对比: 训练指标/生成质量/性能指标）
├── llm_as_judge.py                # LLM 盲评脚本（可选, 需 API key）
├── scripts/
│   ├── convert_ecd_to_chatml.py   # ECD 原始语料 → ChatML JSONL（直接运行即可）
│   ├── download_model.py          # 模型下载（HuggingFace/ModelScope）
│   └── preprocess_data.py         # 通用数据预处理（清洗/去重/格式转换/切分）
├── data/
│   ├── ecd/                       # 电商客服数据（默认训练数据）
│   │   ├── train.jsonl            #   30,000 条
│   │   ├── eval.jsonl             #    4,961 条
│   │   └── test.jsonl             #      992 条
│   ├── alpaca_zh/                 # 通用中文指令数据（备选, 43,936 条）
│   ├── sample/                    # 演示数据（3/2/1 条, 验证流程用）
│   └── hf_cache/                  # datasets 下载缓存
├── models/Qwen2.5-0.5B            # 底座模型
├── dataset/E-commerce dataset/    # ECD 原始语料（train/dev/test.txt, 100万行）
├── output/                        # 训练/测试输出（见下方目录规则）
└── docs/工作记录.md               # 完整搭建记录
```

---

## 快速开始

### 1. 训练（默认 ECD 数据，无需任何参数）

```bash
conda activate qwen25
python train.py
```

建议先试水 1 个 epoch（约 30~60 分钟）：

```bash
python train.py --num_epochs 1
```

**输出**（每次运行独立目录，永不覆盖）：

```
output/ecd_train_20260902_120000/
├── checkpoint-1875/          # epoch 1 结束时的 LoRA 权重 (~35MB)
├── checkpoint-3750/          # epoch 2
├── checkpoint-5625/          # epoch 3
├── final/                    # 最终模型（= 最后一个 epoch）
├── train_history.json        # 完整 loss 曲线 / 学习率记录
├── eval_results.json         # 验证集结果
└── training_config.json      # 本次训练的全部参数
```

> checkpoint 目录名是全局步数（约 1875 步/epoch）。想用某个 epoch 的模型测试：
> `python test.py --model_path output/ecd_train_xxx/checkpoint-3750`

### 2. 测试（自动对比微调前 vs 微调后）

```bash
python test.py --test_data_path ./data/ecd/test.jsonl --max_gen_samples 100
```

自动完成：找最新训练结果 → 从 adapter 配置定位基座模型 → 两个模型分别评估 → 打印对比表。

**输出**（`output/ecd_test_{时间戳}/`）：

| 文件 | 内容 |
|------|------|
| `test_results.json` | 全部指标 + 微调前后对比 + 提升幅度 |
| `generations_compare.json` | 每条的 问题 + 标准答案 + 微调前回复 + 微调后回复（人工抽查用） |

**对比表示例**：

```
指标                        微调前    微调后    变化      说明
eval_loss                   ...                          ↓好 | 预测损失, 越低预测越准
eval_mean_token_accuracy    ...                          ↑好 | 下一词预测准确率
exact_match / bleu / rouge  ...                          ↑好 | 生成与标准答案贴合度
ttft_ms                     ...                          ↓好 | 首 token 延迟(毫秒)
prefill_tokens_per_sec      ...                          ↑好 | 输入处理速度, token/秒
decode_tokens_per_sec       ...                          ↑好 | 输出生成速度, token/秒
```

### 3. LLM 盲评（可选，需要 API key）

```bash
set LLM_API_KEY=sk-xxx
python llm_as_judge.py                # 默认阿里云 DashScope + qwen-max 当评审
```

随机打乱 A/B 位置盲评微调前后回复，按 **相关性/有帮助性/事实性/流畅性** 四维度打分（1-5），输出平均分和胜率到 `llm_judge_results.json`。

---

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `./models/Qwen2.5-0.5B` | 底座模型 |
| `--data_path` / `--eval_data_path` | `./data/ecd/train.jsonl` / `eval.jsonl` | ECD 数据 |
| `--num_epochs` | 3 | 看 eval_loss 回升则降为 2 |
| `--batch_size` × `--gradient_accumulation_steps` | 8 × 2 | 有效 batch 16，显存约 3~4GB，OOM 则降到 4 |
| `--learning_rate` | 2e-4 | LoRA 经典区间 1e-4~3e-4 |
| `--max_seq_length` | 1024 | ECD 最长样本约 1000 token，全覆盖且省显存 |
| `--lora_r / alpha / dropout` | 16 / 32 / 0.05 | 客服单一风格任务 r=16 足够 |
| `--fp16` / `--bf16` | True / False | **GTX 1660 SUPER 不支持 bf16，勿开** |
| `--save_strategy` | epoch | 每个 epoch 保留一份权重 |
| `--save_total_limit` | 不限 | LoRA 检查点每个仅 ~35MB |
| `--logging_steps / --eval_steps` | 50 / 500 | 总步数约 5600，每轮验证 3~4 次 |
| `--use_qlora` | False | 0.5B fp16 仅 1GB，无需量化；换 3B+ 模型时开启 |

**训练时长预估**：约 1.5~3 小时（3 epochs / 3 万条 / GTX 1660 SUPER）。

---

## 评估体系（三层）

| 层次 | 指标 | 视角 |
|------|------|------|
| 训练指标 | eval_loss ↓ / token_accuracy ↑ / entropy | 模型内部收敛性 |
| 生成质量 | exact_match / BLEU / ROUGE-1 / ROUGE-L ↑ | 与标准答案贴合度 |
| 性能指标 | TTFT ↓ / prefill & decode 吞吐 ↑ | 部署延迟与速度 |
| LLM 盲评 | 四维度打分 + 胜率 | 业务质量（可选） |
| 人工抽查 | generations_compare.json | 业务视角 |

---

## 数据说明

**ECD 电商客服数据**（默认）：源自淘宝真实客服对话。原始格式为检索式（`label \t 对话轮次 \t 候选回复`），已由 `scripts/convert_ecd_to_chatml.py` 转换为生成式 ChatML：

- 只保留 `label=1` 正样本，丢弃负样本
- 去除分词空格恢复连续中文
- 奇数位=user（顾客）、偶数位=assistant（客服）
- 自动附加电商客服 system prompt
- 支持随机抽样（`--max_samples`，默认 30000）

重新转换 / 扩大数据量：

```bash
python scripts/convert_ecd_to_chatml.py                          # 默认: 抽 3 万条
python scripts/convert_ecd_to_chatml.py --max_samples 100000     # 抽 10 万条
```

**换数据训练**：

```bash
python train.py --data_path ./data/alpaca_zh/train.jsonl --eval_data_path ./data/alpaca_zh/eval.jsonl
```

---

## 常见问题

**Q: 报错 bf16 / mat1 and mat2 dtype 不一致？**
GTX 1660 SUPER（Turing）不支持 bf16，保持 `--fp16=True --bf16=False`（默认已如此）。

**Q: 显存不足（OOM）？**
降低 `--batch_size 4` 或开启 `--use_qlora`（4-bit 量化加载）。

**Q: HuggingFace 下载失败/卡住？**
使用镜像并禁用 Xet：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

**Q: 训练结果会被覆盖吗？**
不会。每次训练/测试都写入 `output/{数据集名}_{train|test}_{时间戳}/` 独立目录。

**Q: transformers 5.x 的 `torch_dtype` 弃用警告？**
无害，可忽略；如遇兼容问题可降级 `pip install "transformers>=4.35,<5"`（注意 warmup_ratio/max_length 等 API 需同步回退）。

---

## License

MIT License
