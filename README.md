# Qwen2.5 电商客服微调（ECD · LoRA）

基于**淘宝真实客服对话语料（ECD）**微调 Qwen2.5-0.5B 的端到端项目：数据转换 → LoRA 训练 → 三层评估（训练指标 / 生成质量 / LLM 盲评）→ 权重合并 → 对话部署。所有运行结果按时间戳隔离保存，互不覆盖。

> 本项目在 [Luoyifeiandxs/qwen2.5-finetune](https://github.com/Luoyifeiandxs/qwen2.5-finetune)（MIT）基础上深度复现与二次开发。新增内容见「六、我的改进」。

## 一、成果概览

| 项目 | 结果 |
|---|---|
| 训练数据 | ECD 电商客服对话：train 10,000 条 / eval 4,788 条 / test 952 条 |
| 底座模型 | Qwen2.5-0.5B（4.94 亿参数，fp16 加载约 1GB 显存） |
| 训练方式 | LoRA（r=16, α=32），可训练参数 879 万（占全量 1.75%） |
| 最优配置 | **2 epochs**（3-epoch 出现过拟合，判据见「五、实验结论」） |
| 测试集指标 | eval_loss 6.27→**2.38**；下一词准确率 0.206→**0.526**；BLEU 0.0015→**0.108** |
| LLM 盲评 | 微调后胜率 **92.4%**（四维度：相关性 2.41 / 有帮助性 2.13 / 事实性 2.54 / 流畅性 3.13，满分 5） |
| 部署产物 | LoRA 合并后的完整模型（float16，约 1GB），可直接对话 |

> 模型定位：**短句话术型客服**——砍价、快递、库存、简单售后等高频场景应答合格；复杂多轮售后与训练集外问题会答偏/编造（0.5B + 短对话数据的预期内表现，详见「八、评估体系」）。

## 二、架构链路

```
ECD 原始语料（label \t 对话轮次 \t 候选回复）
  → scripts/convert_ecd_to_chatml.py 转换为 ChatML JSONL
  → train.py  LoRA 微调（按 epoch 存 checkpoint + train_history.json）
  → test.py   三层评估（微调前 vs 微调后：指标对比 + 100 条生成对照）
  → llm_as_judge.py  大模型盲评（可选，四维度打分 + 胜率）
  → merge_model.py   合并 LoRA 权重 → 完整模型
  → chat.py   命令行对话体验
```

## 三、项目结构

```
qwen2.5-finetune/
├── train.py                       # 训练（LoRA/QLoRA 可选，按 epoch 保存检查点）
├── test.py                        # 评估（微调前后对比：指标/生成质量/性能）※已加全中文注释
├── llm_as_judge.py                # LLM 盲评（OpenAI 兼容接口，支持硅基流动/DashScope）
├── merge_model.py                 # LoRA → 完整模型合并
├── chat.py                        # 命令行对话（多模型切换、系统提示）
├── scripts/
│   ├── convert_ecd_to_chatml.py   # ECD 原始语料 → ChatML JSONL
│   ├── download_model.py          # 模型下载（HuggingFace / ModelScope 魔搭）
│   └── preprocess_data.py         # 通用数据预处理
├── data/
│   ├── ecd/                       # 电商客服数据（train_10k.jsonl / eval.jsonl / test.jsonl）
│   ├── ecd_food/                  # 食品品类数据（备选）
│   └── sample/                    # 演示数据（3/2/1 条，冒烟测试用）
├── models/                        # 底座模型 + 合并后模型（gitignore 排除）
├── output/                        # 训练/测试结果（时间戳隔离，gitignore 排除）
└── 微调学习笔记.md                 # 全流程学习笔记（踩坑/指标详解/结果记录）
```

## 四、快速开始

### 0. 环境（Windows + conda 已验证）

| 项 | 值 |
|---|---|
| conda 环境 | `qwen25`（Python 3.10） |
| PyTorch | 2.5.1+cu121（CUDA 可用） |
| 关键依赖 | transformers 5.16.1 / trl 1.12.0 / peft 0.20.0 / bitsandbytes 0.50.2 / datasets 5.0.1 |
| 显卡 | RTX 3050 Laptop **4GB**（本项目全程 fp16，未用 bf16） |

> ⚠️ Windows 坑：conda 4.x 的 PowerShell 钩子与 PowerShell 7+ 不兼容（`conda activate` 报错），项目命令统一用完整解释器路径：
> ```powershell
> D:\miniconda\envs\qwen25\python.exe train.py ...
> ```

### 1. 数据准备（仓库已带转换后的数据，可跳过）

```bash
# 从 ModelScope 魔搭下载底座模型
python scripts/download_model.py --source modelscope

# 重新转换 ECD 原始语料（默认抽 3 万条）
python scripts/convert_ecd_to_chatml.py
```

### 2. 训练（推荐 2 epochs）

```bash
# 冒烟测试：3 条数据验证全流程
python train.py --data_path ./data/sample/train.json --eval_data_path ./data/sample/eval.json --num_epochs 1

# 正式训练（本项目实测：2 epochs / RTX 3050 4GB / batch 4 / 约 3 小时）
python train.py --num_epochs 2 --batch_size 4
```

**输出**（`output/ecd_train_{时间戳}/`）：`checkpoint-{步数}/`（LoRA，~35MB）、`final/`、`train_history.json`、`eval_results.json`、`training_config.json`。

### 3. 评估（自动对比微调前 vs 微调后）

```bash
python test.py --model_path output/ecd_train_xxx/checkpoint-5000 --test_data_path ./data/ecd/test.jsonl --max_gen_samples 100
```

输出 `output/ecd_test_{时间戳}/`：`test_results.json`（全部指标对比）+ `generations_compare.json`（100 条 问题/标准答案/两模型回复，人工抽查用）。

### 4. LLM 盲评（可选，需 OpenAI 兼容 API key）

```bash
python llm_as_judge.py --api_base https://api.siliconflow.cn/v1 --api_key sk-xxx --judge_model Qwen/Qwen2.5-72B-Instruct --max_samples 100
```

自动读取最新 `generations_compare.json`，随机打乱 A/B 后按 **相关性 / 有帮助性 / 事实性 / 流畅性**（1-5 分）评审并判胜者，输出 `llm_judge_results.json`。

### 5. 合并权重（部署前必做）

```bash
python merge_model.py --model_path output/ecd_train_xxx/checkpoint-5000
```

⚠️ **必须显式传 `--model_path`**：脚本默认值是作者电脑上的路径（非自动查找），不带参数会报 `Not a LoRA adapter`。合并原理 `W' = W + B·A`（数学无损），几秒完成，输出 `models/{run名}-merged/`。

### 6. 对话体验

```bash
python chat.py --model_path ./models/ecd_train_xxx-merged
```

## 五、实验结论（2-epoch 最优，3-epoch 过拟合）

| 观察点 | Epoch 1 → 2 | Epoch 3 |
|---|---|---|
| eval_loss | 2.523 → **2.349**（step 5000 全周期最低） | step 5250 跳涨至 2.548，后平台震荡 2.50~2.52 |
| 验证准确率 | 0.492 → 0.527（持续上升） | 卡死 0.526 |
| 训练 loss | 2.72 → 1.94 | 猛降至 ~1.3（开始背训练集） |
| 验证熵 | 2.45 → 2.10 | 骤降至 1.60（过度自信） |

**过拟合判据**：训练 loss 继续降 + 验证 loss 回升 + 验证熵骤降 = 模型开始背训练集而非学规律。**正式成果选用 checkpoint-5000（2-epoch）**，并在独立测试集（952 条，模型未见过）上验证：所有质量指标优于 1-epoch。

## 六、我的改进（相对上游）

1. **3-epoch 过拟合实验**：完整训练 3 轮并记录每步指标，用数据驱动选定最优 checkpoint，而非默认用 final
2. **独立测试集验证 + LLM 盲评体系**：搭建"指标对比 + 大模型四维盲评"双通道评估，胜率 92.4% 基线存档，可复现可对比
3. **test.py 全中文注释**：逻辑零改动（AST 对比验证），逐函数解释"在做什么、为什么"
4. **踩坑记录沉淀**：torch 清华源为 CPU 版、conda×PowerShell 7 不兼容、merge_model 默认路径、LoRA 推理开销等（见下）

## 七、踩坑记录

| 问题 | 原因 | 解决 |
|---|---|---|
| `torch.cuda.is_available()` 为 False | 清华 pip 源默认 torch 为 CPU 版 | 直接下载阿里云 cu121 wheel 安装 |
| `conda activate` 报 `Invoke-Expression: Missing argument` | conda 4.12 钩子与 PowerShell 7+ 不兼容 | 用 `D:\miniconda\envs\qwen25\python.exe` 完整路径跑命令 |
| `merge_model.py` 报 `Not a LoRA adapter` | 默认 `--model_path` 是作者电脑路径 | 显式传 `--model_path` |
| `chat.py` 报 transformers ImportError | 误用系统 Python 3.11（非 qwen25 环境） | 统一用 qwen25 解释器 |
| OOM（显存不足） | batch 过大 / 模型过大 | 降 `--batch_size 4` 或开 `--use_qlora` |
| bf16 报错 | RTX 3050/1660（Turing/Ampere 前代）不支持 bf16 | 保持 `--fp16 True --bf16 False` |

## 八、评估体系（三层）

| 层次 | 指标 | 视角 |
|---|---|---|
| 训练指标 | eval_loss ↓ / token_accuracy ↑ / entropy ↓ | 模型内部收敛性 |
| 生成质量 | exact_match / BLEU / ROUGE-1 / ROUGE-L ↑ | 与标准答案贴合度 |
| 性能指标 | TTFT ↓ / prefill & decode 吞吐 ↑ | 部署延迟与速度（LoRA 加载有开销，合并后恢复原生速度） |
| LLM 盲评 | 四维度打分 + 胜率 | 业务质量（可选） |
| 人工抽查 | generations_compare.json | 业务视角 |

盲评判卷规律：简短确认类（"看见订单了么"→"看到了哦"）4 分；信息咨询类 3 分（答得出但生硬）；复杂售后/抱怨类 1-2 分（只能挤出安抚短句）。**有帮助性（2.13）最弱** = 后续提升的第一线索。

## 九、后续路线

- [ ] 全量 3 万条数据训练（`data/ecd/train.jsonl`），覆盖更多盲区
- [ ] Qwen2.5-1.5B + QLoRA（4-bit 量化加载，4GB 显存可跑）
- [ ] 长回复训练（针对复杂售后场景）或回答信息增强
- [ ] FastAPI 服务化部署 + 监控日志

## 十、致谢与协议

- 上游项目：[Luoyifeiandxs/qwen2.5-finetune](https://github.com/Luoyifeiandxs/qwen2.5-finetune)（MIT）
- 训练数据：ECD 电商客服语料（淘宝真实对话，版权归原作者）
- 本项目遵循 MIT License
