"""
任务特定指标
包括代码评估、幻觉检测、拒绝率等
"""

import re
import math
from typing import List, Dict, Any
from collections import Counter
import numpy as np


class TaskMetrics:
    """任务特定指标"""

    def code_eval(
        self,
        predictions: List[str],
        references: List[str],
        timeout: float = 5.0
    ) -> float:
        """
        计算代码评估分数 (pass@k)
        
        Args:
            predictions: 预测的代码
            references: 参考代码
            timeout: 执行超时时间
        
        Returns:
            pass@1 分数
        """
        correct = 0
        total = len(predictions)

        for pred, ref in zip(predictions, references):
            if self._code_matches(pred, ref):
                correct += 1

        return correct / total if total > 0 else 0.0

    def _code_matches(self, pred: str, ref: str) -> bool:
        """检查代码是否匹配"""
        # 简化版：检查关键代码结构
        try:
            # 提取函数名
            pred_funcs = set(re.findall(r'def\s+(\w+)', pred))
            ref_funcs = set(re.findall(r'def\s+(\w+)', ref))

            # 提取类名
            pred_classes = set(re.findall(r'class\s+(\w+)', pred))
            ref_classes = set(re.findall(r'class\s+(\w+)', ref))

            # 检查关键结构是否匹配
            if pred_funcs == ref_funcs and pred_classes == ref_classes:
                return True

            # 检查返回语句
            pred_returns = set(re.findall(r'return\s+(.+)', pred))
            ref_returns = set(re.findall(r'return\s+(.+)', ref))

            if pred_returns == ref_returns:
                return True

        except Exception:
            pass

        return False

    def hallucination_rate(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        计算幻觉率
        
        检测预测中的不确定表达和事实不一致
        """
        hallucination_indicators = [
            r"I don't know",
            r"I'm not sure",
            r"I cannot",
            r"I'm unable",
            r"As an AI",
            r"I don't have access",
            r"I cannot verify",
            r"我不确定",
            r"我不知道",
            r"无法确认",
            r"作为AI",
        ]

        hallucination_count = 0

        for pred, ref in zip(predictions, references):
            has_hallucination = False

            # 检查不确定表达
            for pattern in hallucination_indicators:
                if re.search(pattern, pred, re.IGNORECASE):
                    has_hallucination = True
                    break

            # 检查事实一致性
            if not has_hallucination:
                pred_facts = set(re.findall(r'\d+\.?\d*', pred))
                ref_facts = set(re.findall(r'\d+\.?\d*', ref))

                if len(pred_facts) > 0:
                    unsupported = pred_facts - ref_facts
                    if len(unsupported) / len(pred_facts) > 0.5:
                        has_hallucination = True

            if has_hallucination:
                hallucination_count += 1

        return hallucination_count / len(predictions) if predictions else 0.0

    def refusal_rate(
        self,
        predictions: List[str]
    ) -> float:
        """
        计算拒绝率
        
        检测模型拒绝回答的比例
        """
        refusal_patterns = [
            r"I cannot",
            r"I can't",
            r"I'm not able",
            r"I don't have",
            r"I'm unable",
            r"我无法",
            r"我不能",
            r"我没有能力",
            r"抱歉",
            r"sorry",
        ]

        refusal_count = 0

        for pred in predictions:
            for pattern in refusal_patterns:
                if re.search(pattern, pred, re.IGNORECASE):
                    refusal_count += 1
                    break

        return refusal_count / len(predictions) if predictions else 0.0

    def relevance_score(
        self,
        predictions: List[str],
        prompts: List[str]
    ) -> float:
        """
        计算相关性分数
        
        基于预测与问题的关键词重叠
        """
        scores = []

        for pred, prompt in zip(predictions, prompts):
            # 提取关键词
            prompt_words = set(self._extract_keywords(prompt))
            pred_words = set(self._extract_keywords(pred))

            if len(prompt_words) == 0:
                scores.append(0.5)
                continue

            # 计算关键词覆盖
            overlap = len(prompt_words & pred_words)
            coverage = overlap / len(prompt_words)

            scores.append(min(coverage * 1.5, 1.0))  # 放大到 0-1

        return np.mean(scores)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 移除停用词
        stop_words = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'don', 'now'
        ])

        words = re.findall(r'\w+', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 1]

    def completeness_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        计算完整性分数
        
        检测预测是否覆盖了参考答案的关键信息
        """
        scores = []

        for pred, ref in zip(predictions, references):
            ref_facts = self._extract_facts(ref)
            pred_facts = self._extract_facts(pred)

            if len(ref_facts) == 0:
                scores.append(1.0)
                continue

            covered = len(pred_facts & ref_facts)
            scores.append(covered / len(ref_facts))

        return np.mean(scores)

    def _extract_facts(self, text: str) -> set:
        """提取事实"""
        facts = set()

        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', text)
        facts.update(numbers)

        # 提取关键名词
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                facts.add(word.lower())

        return facts
