"""
企业级评估引擎
支持异步评估、并行处理、缓存和资源监控
"""

import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .model_manager import ModelManager
from .cache_manager import CacheManager
from .resource_monitor import ResourceMonitor
from ..metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """评估结果数据类"""
    metric_name: str
    value: float
    confidence_interval: Optional[tuple] = None
    sample_size: int = 0
    computation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """完整评估报告"""
    model_name: str
    timestamp: str
    dataset_name: str
    metrics: Dict[str, EvalResult]
    predictions: Optional[List[Dict]] = None
    resource_usage: Optional[Dict] = None
    comparison_with_baseline: Optional[Dict] = None


class EvalEngine:
    """企业级评估引擎"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model_manager = ModelManager(self.config["model"])
        self.cache_manager = CacheManager(self.config.get("cache", {}))
        self.resource_monitor = ResourceMonitor()
        self.metrics_calculator = MetricsCalculator(self.config.get("metrics", {}))

    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    async def run_evaluation(
        self,
        model_path: str,
        dataset_name: str,
        metrics: List[str],
        dataset: Optional[List[Dict]] = None,
        **kwargs
    ) -> EvalReport:
        """运行完整评估流程"""
        logger.info(f"Starting evaluation: model={model_path}, dataset={dataset_name}")

        with self.resource_monitor.track():
            # 1. 加载模型
            model, tokenizer = self.model_manager.load_model(model_path)

            # 2. 加载数据集
            if dataset is None:
                dataset = self._load_dataset(dataset_name)

            # 3. 检查缓存
            cache_key = self._get_cache_key(model_path, dataset_name, metrics)
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info("Using cached evaluation results")
                return cached_result

            # 4. 生成预测
            predictions = await self._generate_predictions(model, tokenizer, dataset)

            # 5. 计算指标
            metric_results = await self._compute_metrics(predictions, dataset, metrics)

            # 6. 生成报告
            report = EvalReport(
                model_name=model_path,
                timestamp=datetime.now().isoformat(),
                dataset_name=dataset_name,
                metrics=metric_results,
                predictions=predictions,
                resource_usage=self.resource_monitor.get_usage()
            )

            # 7. 缓存结果
            await self.cache_manager.set(cache_key, report)

        logger.info(f"Evaluation completed in {report.resource_usage['duration_seconds']:.2f}s")
        return report

    def _load_dataset(self, dataset_name: str) -> List[Dict]:
        """加载数据集"""
        from ..datasets import DatasetLoader

        loader = DatasetLoader(self.config.get("datasets", []))
        return loader.load(dataset_name)

    def _get_cache_key(
        self,
        model_path: str,
        dataset_name: str,
        metrics: List[str]
    ) -> str:
        """生成缓存键"""
        import hashlib
        key = f"{model_path}:{dataset_name}:{','.join(sorted(metrics))}"
        return hashlib.md5(key.encode()).hexdigest()

    async def _generate_predictions(
        self,
        model,
        tokenizer,
        dataset: List[Dict]
    ) -> List[Dict]:
        """批量生成预测"""
        predictions = []
        batch_size = self.config.get("generation", {}).get("batch_size", 8)

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_predictions = await self._generate_batch(
                model, tokenizer, batch
            )
            predictions.extend(batch_predictions)

            if i % self.config.get("monitoring", {}).get("log_interval", 10) == 0:
                logger.info(f"Generated {i + len(batch)}/{len(dataset)} predictions")

        return predictions

    async def _generate_batch(
        self,
        model,
        tokenizer,
        batch: List[Dict]
    ) -> List[Dict]:
        """生成单个批次的预测"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._generate_single,
                    model,
                    tokenizer,
                    item
                )
                for item in batch
            ]
            results = await asyncio.gather(*tasks)

        return results

    def _generate_single(self, model, tokenizer, item: Dict) -> Dict:
        """生成单条预测"""
        inputs = tokenizer(
            item.get("prompt", ""),
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("generation", {}).get("max_length", 4096)
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.get("generation", {}).get("max_new_tokens", 1024),
                temperature=self.config.get("generation", {}).get("temperature", 0.7),
                top_p=self.config.get("generation", {}).get("top_p", 0.9),
                top_k=self.config.get("generation", {}).get("top_k", 50),
                do_sample=self.config.get("generation", {}).get("do_sample", True),
                repetition_penalty=self.config.get("generation", {}).get("repetition_penalty", 1.1),
                num_beams=self.config.get("generation", {}).get("num_beams", 1),
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "id": item.get("id", str(hash(item.get("prompt", "")))),
            "prompt": item.get("prompt", ""),
            "prediction": prediction,
            "reference": item.get("reference", item.get("answer", "")),
            "metadata": item.get("metadata", {})
        }

    async def _compute_metrics(
        self,
        predictions: List[Dict],
        dataset: List[Dict],
        metrics: List[str]
    ) -> Dict[str, EvalResult]:
        """计算所有指标"""
        results = {}

        for metric_name in metrics:
            start_time = time.time()

            try:
                result = self.metrics_calculator.compute(
                    metric_name, predictions, dataset
                )
                result.computation_time = time.time() - start_time
                results[metric_name] = result

            except Exception as e:
                logger.error(f"Error computing metric {metric_name}: {e}")
                results[metric_name] = EvalResult(
                    metric_name=metric_name,
                    value=float('nan'),
                    metadata={"error": str(e)}
                )

        return results

    def compare_models(
        self,
        results1: EvalReport,
        results2: EvalReport
    ) -> Dict[str, Any]:
        """比较两个模型的评估结果"""
        comparison = {
            "model1": results1.model_name,
            "model2": results2.model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }

        common_metrics = set(results1.metrics.keys()) & set(results2.metrics.keys())

        for metric in common_metrics:
            val1 = results1.metrics[metric].value
            val2 = results2.metrics[metric].value

            comparison["metrics"][metric] = {
                "model1_value": val1,
                "model2_value": val2,
                "difference": val2 - val1,
                "relative_change": (val2 - val1) / val1 if val1 != 0 else float('inf'),
                "improvement": val2 > val1
            }

        return comparison
