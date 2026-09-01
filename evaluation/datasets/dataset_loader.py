"""
数据集加载器
支持多种数据格式和数据源
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    path: str
    format: str = "chatml"
    max_samples: Optional[int] = None
    subset: Optional[str] = None
    split: str = "test"
    cache_dir: Optional[str] = None


class DatasetLoader:
    """数据集加载器"""

    def __init__(self, configs: List[Dict[str, Any]]):
        self.configs = {
            c["name"]: DatasetConfig(**c) for c in configs
        }
        self.loaded_datasets = {}

    def load(
        self,
        name: str,
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """加载数据集"""
        if name in self.loaded_datasets:
            return self.loaded_datasets[name]

        if name not in self.configs:
            raise ValueError(f"Unknown dataset: {name}")

        config = self.configs[name]
        logger.info(f"Loading dataset: {name} from {config.path}")

        # 判断数据源类型
        if config.path.startswith("./") or config.path.startswith("/"):
            dataset = self._load_local(config)
        else:
            dataset = self._load_hub(config)

        # 限制样本数
        limit = max_samples or config.max_samples
        if limit and len(dataset) > limit:
            dataset = dataset[:limit]

        # 转换格式
        dataset = self._convert_format(dataset, config.format)

        self.loaded_datasets[name] = dataset
        logger.info(f"Loaded {len(dataset)} samples from {name}")

        return dataset

    def _load_local(self, config: DatasetConfig) -> List[Dict]:
        """加载本地数据集"""
        path = Path(config.path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if path.suffix == ".json":
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # 支持 {"data": [...]} 格式
                    for key in ["data", "samples", "examples"]:
                        if key in data:
                            return data[key]
                    return [data]

        elif path.suffix == ".jsonl":
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data

        elif path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict('records')

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_hub(self, config: DatasetConfig) -> List[Dict]:
        """从 HuggingFace Hub 加载数据集"""
        try:
            from datasets import load_dataset

            dataset = load_dataset(
                config.path,
                config.subset,
                split=config.split,
                cache_dir=config.cache_dir
            )

            # 转换为列表
            data = []
            for item in dataset:
                data.append(dict(item))

            return data

        except Exception as e:
            logger.error(f"Failed to load dataset from hub: {e}")
            raise

    def _convert_format(
        self,
        data: List[Dict],
        format: str
    ) -> List[Dict]:
        """转换数据格式"""
        if format == "chatml":
            return self._to_chatml(data)
        elif format == "alpaca":
            return self._to_alpaca(data)
        elif format == "sharegpt":
            return self._to_sharegpt(data)
        else:
            return data

    def _to_chatml(self, data: List[Dict]) -> List[Dict]:
        """转换为 ChatML 格式"""
        converted = []

        for item in data:
            # 如果已经是标准格式
            if "prompt" in item and ("reference" in item or "answer" in item):
                converted.append(item)
                continue

            # 处理 messages 格式
            if "messages" in item:
                converted.append({
                    "prompt": self._messages_to_prompt(item["messages"]),
                    "reference": self._extract_assistant_response(item["messages"]),
                    "metadata": item.get("metadata", {})
                })
                continue

            # 处理 instruction/input/output 格式
            if "instruction" in item:
                prompt = item["instruction"]
                if "input" in item and item["input"]:
                    prompt += f"\n{item['input']}"

                converted.append({
                    "prompt": prompt,
                    "reference": item.get("output", ""),
                    "metadata": item.get("metadata", {})
                })
                continue

            # 尝试通用字段
            converted.append({
                "prompt": item.get("question", item.get("query", "")),
                "reference": item.get("answer", item.get("response", "")),
                "metadata": {k: v for k, v in item.items()
                           if k not in ["question", "query", "answer", "response"]}
            })

        return converted

    def _to_alpaca(self, data: List[Dict]) -> List[Dict]:
        """转换为 Alpaca 格式"""
        converted = []

        for item in data:
            if "instruction" in item:
                prompt = item["instruction"]
                if "input" in item and item["input"]:
                    prompt += f"\n{item['input']}"

                converted.append({
                    "prompt": prompt,
                    "reference": item.get("output", ""),
                    "metadata": item.get("metadata", {})
                })
            elif "prompt" in item:
                converted.append(item)

        return converted

    def _to_sharegpt(self, data: List[Dict]) -> List[Dict]:
        """转换为 ShareGPT 格式"""
        converted = []

        for item in data:
            if "conversations" in item:
                conversations = item["conversations"]
                prompt = ""
                reference = ""

                for turn in conversations:
                    if turn.get("from") == "human":
                        prompt = turn.get("value", "")
                    elif turn.get("from") == "gpt":
                        reference = turn.get("value", "")

                converted.append({
                    "prompt": prompt,
                    "reference": reference,
                    "metadata": item.get("metadata", {})
                })

        return converted

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """将消息列表转换为提示词"""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "system":
                parts.append(f"System: {content}")

        return "\n".join(parts)

    def _extract_assistant_response(self, messages: List[Dict]) -> str:
        """从消息列表中提取助手回复"""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def get_info(self, name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        if name in self.configs:
            config = self.configs[name]
            info = {
                "name": config.name,
                "path": config.path,
                "format": config.format,
                "max_samples": config.max_samples,
            }

            if name in self.loaded_datasets:
                info["loaded_samples"] = len(self.loaded_datasets[name])

            return info
        return {}
