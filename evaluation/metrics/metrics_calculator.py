"""
指标计算器统一入口
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from .base_metrics import BaseMetrics
from .generation_metrics import GenerationMetrics
from .task_metrics import TaskMetrics
from .statistical_metrics import StatisticalMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """评估结果"""
    metric_name: str
    value: float
    confidence_interval: Optional[tuple] = None
    sample_size: int = 0
    computation_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MetricsCalculator:
    """指标计算器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_metrics = BaseMetrics()
        self.generation_metrics = GenerationMetrics()
        self.task_metrics = TaskMetrics()
        self.statistical_metrics = StatisticalMetrics()

    def compute(
        self,
        metric_name: str,
        predictions: List[Dict],
        dataset: List[Dict]
    ) -> EvalResult:
        """计算指定指标"""
        # 提取预测和参考
        preds = [p.get("prediction", "") for p in predictions]
        refs = [p.get("reference", "") for p in predictions]
        prompts = [p.get("prompt", "") for p in predictions]

        # 根据指标名称调用相应方法
        metric_map = {
            "perplexity": lambda: self._compute_perplexity(preds, refs),
            "accuracy": lambda: self.base_metrics.accuracy(preds, refs),
            "f1_score": lambda: self.base_metrics.f1_score(preds, refs),
            "bleu": lambda: self.generation_metrics.bleu_score(preds, refs),
            "rouge_l": lambda: self.generation_metrics.rouge_scores(preds, refs).get("rougeL_fmeasure", 0),
            "meteor": lambda: self.generation_metrics.meteor_score(preds, refs),
            "code_eval": lambda: self.task_metrics.code_eval(preds, refs),
            "hallucination_rate": lambda: self.task_metrics.hallucination_rate(preds, refs),
            "refusal_rate": lambda: self.task_metrics.refusal_rate(preds),
            "latency_p50": lambda: self._compute_latency(preds, percentile=50),
            "latency_p95": lambda: self._compute_latency(preds, percentile=95),
            "latency_p99": lambda: self._compute_latency(preds, percentile=99),
        }

        if metric_name not in metric_map:
            raise ValueError(f"Unknown metric: {metric_name}")

        try:
            value = metric_map[metric_name]()

            # 计算置信区间
            ci = self.statistical_metrics.confidence_interval(
                [value] if isinstance(value, (int, float)) else value,
                confidence=0.95
            ) if isinstance(value, (int, float)) else None

            return EvalResult(
                metric_name=metric_name,
                value=float(value),
                confidence_interval=ci,
                sample_size=len(preds)
            )
        except Exception as e:
            logger.error(f"Error computing {metric_name}: {e}")
            return EvalResult(
                metric_name=metric_name,
                value=float('nan'),
                metadata={"error": str(e)}
            )

    def _compute_perplexity(self, preds: List[str], refs: List[str]) -> float:
        """计算困惑度（简化版）"""
        # 这里使用一个简化的困惑度计算
        # 实际应该使用模型的logits
        from collections import Counter
        import math

        all_tokens = []
        for pred in preds:
            all_tokens.extend(pred.split())

        token_counts = Counter(all_tokens)
        total = sum(token_counts.values())

        entropy = -sum(
            (count/total) * math.log2(count/total)
            for count in token_counts.values()
        )

        return 2 ** entropy

    def _compute_latency(self, preds: List[str], percentile: int = 50) -> float:
        """计算延迟（模拟）"""
        # 实际应该在推理时测量
        # 这里用文本长度作为代理
        lengths = [len(p.split()) for p in preds]
        return np.percentile(lengths, percentile)

    def compute_all(
        self,
        predictions: List[Dict],
        dataset: List[Dict]
    ) -> Dict[str, EvalResult]:
        """计算所有配置的指标"""
        results = {}

        metric_categories = self.config.get("automated", [])
        for metric_name in metric_categories:
            results[metric_name] = self.compute(metric_name, predictions, dataset)

        return results
