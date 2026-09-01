"""
基础指标计算
"""

import math
from typing import List, Dict, Any
from collections import Counter
import numpy as np


class BaseMetrics:
    """基础评估指标"""

    @staticmethod
    def perplexity(
        logits: np.ndarray,
        labels: np.ndarray,
        ignore_index: int = -100
    ) -> float:
        """
        计算困惑度
        
        Args:
            logits: 模型输出的logits, shape: (batch, seq_len, vocab_size)
            labels: 真实标签, shape: (batch, seq_len)
            ignore_index: 忽略的标签索引
        
        Returns:
            困惑度值
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 计算交叉熵
        loss_fct = lambda x, y: -np.mean(
            np.log(np.softmax(x, axis=-1)[np.arange(len(y)), y] + 1e-10)
        )

        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1)
        )

        return math.exp(loss)

    @staticmethod
    def accuracy(
        predictions: List[str],
        references: List[str],
        case_sensitive: bool = False
    ) -> float:
        """
        计算准确率
        
        Args:
            predictions: 预测列表
            references: 参考答案列表
            case_sensitive: 是否区分大小写
        
        Returns:
            准确率 (0-1)
        """
        if not case_sensitive:
            predictions = [p.lower().strip() for p in predictions]
            references = [r.lower().strip() for r in references]

        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        return correct / len(predictions) if predictions else 0.0

    @staticmethod
    def f1_score(
        predictions: List[str],
        references: List[str],
        average: str = "macro"
    ) -> float:
        """
        计算 F1 分数
        
        Args:
            predictions: 预测列表
            references: 参考答案列表
            average: 平均方式 ("macro" 或 "micro")
        
        Returns:
            F1 分数 (0-1)
        """
        if average == "macro":
            f1_scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = set(pred.split())
                ref_tokens = set(ref.split())

                if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                    f1_scores.append(1.0)
                    continue

                if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                    f1_scores.append(0.0)
                    continue

                common = pred_tokens & ref_tokens
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(ref_tokens)

                if precision + recall == 0:
                    f1_scores.append(0.0)
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f1_scores.append(f1)

            return np.mean(f1_scores) if f1_scores else 0.0

        elif average == "micro":
            all_pred_tokens = Counter()
            all_ref_tokens = Counter()
            for pred, ref in zip(predictions, references):
                all_pred_tokens.update(pred.split())
                all_ref_tokens.update(ref.split())

            common = sum((all_pred_tokens & all_ref_tokens).values())
            total_pred = sum(all_pred_tokens.values())
            total_ref = sum(all_ref_tokens.values())

            precision = common / total_pred if total_pred > 0 else 0
            recall = common / total_ref if total_ref > 0 else 0

            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)

        return 0.0

    @staticmethod
    def precision(
        predictions: List[str],
        references: List[str]
    ) -> float:
        """计算精确率"""
        precisions = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())

            if len(pred_tokens) == 0:
                precisions.append(0.0)
                continue

            common = pred_tokens & ref_tokens
            precisions.append(len(common) / len(pred_tokens))

        return np.mean(precisions) if precisions else 0.0

    @staticmethod
    def recall(
        predictions: List[str],
        references: List[str]
    ) -> float:
        """计算召回率"""
        recalls = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())

            if len(ref_tokens) == 0:
                recalls.append(0.0)
                continue

            common = pred_tokens & ref_tokens
            recalls.append(len(common) / len(ref_tokens))

        return np.mean(recalls) if recalls else 0.0

    @staticmethod
    def exact_match(
        predictions: List[str],
        references: List[str]
    ) -> float:
        """计算精确匹配率"""
        return BaseMetrics.accuracy(predictions, references, case_sensitive=True)

    @staticmethod
    def contains_match(
        predictions: List[str],
        references: List[str]
    ) -> float:
        """计算包含匹配率"""
        matches = sum(
            1 for pred, ref in zip(predictions, references)
            if ref.lower() in pred.lower()
        )
        return matches / len(predictions) if predictions else 0.0
