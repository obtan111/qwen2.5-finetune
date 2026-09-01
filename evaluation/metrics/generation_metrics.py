"""
生成质量指标
包括 BLEU, ROUGE, METEOR, 语义相似度等
"""

from typing import List, Dict, Any
from collections import Counter
import numpy as np


class GenerationMetrics:
    """生成质量指标"""

    def __init__(self):
        self._rouge_scorer = None
        self._meteor_scorer = None

    def _get_rouge_scorer(self):
        """懒加载 ROUGE scorer"""
        if self._rouge_scorer is None:
            from rouge_score import rouge_scorer
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=True
            )
        return self._rouge_scorer

    def bleu_score(
        self,
        predictions: List[str],
        references: List[str],
        max_n: int = 4
    ) -> float:
        """
        计算 BLEU 分数
        
        Args:
            predictions: 预测列表
            references: 参考答案列表
            max_n: 最大 n-gram 阶数
        
        Returns:
            BLEU-4 分数 (0-1)
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

            smoothing = SmoothingFunction().method1
            bleu_scores = []

            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]

                # 计算 BLEU-4
                weights = (0.25, 0.25, 0.25, 0.25)
                score = sentence_bleu(
                    ref_tokens,
                    pred_tokens,
                    weights=weights,
                    smoothing_function=smoothing
                )
                bleu_scores.append(score)

            return np.mean(bleu_scores)

        except ImportError:
            # 简化版 BLEU
            return self._simple_bleu(predictions, references)

    def _simple_bleu(self, predictions: List[str], references: List[str]) -> float:
        """简化版 BLEU"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            # 计算 1-gram 到 4-gram 的精确率
            precisions = []
            for n in range(1, 5):
                pred_ngrams = Counter(
                    tuple(pred_tokens[i:i+n])
                    for i in range(len(pred_tokens) - n + 1)
                )
                ref_ngrams = Counter(
                    tuple(ref_tokens[i:i+n])
                    for i in range(len(ref_tokens) - n + 1)
                )

                clipped = sum(
                    min(count, ref_ngrams.get(ngram, 0))
                    for ngram, count in pred_ngrams.items()
                )
                total = sum(pred_ngrams.values())

                precision = clipped / total if total > 0 else 0
                precisions.append(precision)

            # 几何平均
            if all(p > 0 for p in precisions):
                score = np.exp(np.mean(np.log(precisions)))
            else:
                score = 0

            scores.append(score)

        return np.mean(scores)

    def rouge_scores(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算 ROUGE 分数
        
        Returns:
            包含 rouge1, rouge2, rougeL 的 precision, recall, fmeasure
        """
        scorer = self._get_rouge_scorer()

        all_scores = {
            'rouge1_precision': [],
            'rouge1_recall': [],
            'rouge1_fmeasure': [],
            'rouge2_precision': [],
            'rouge2_recall': [],
            'rouge2_fmeasure': [],
            'rougeL_precision': [],
            'rougeL_recall': [],
            'rougeL_fmeasure': [],
        }

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)

            for metric in ['rouge1', 'rouge2', 'rougeL']:
                all_scores[f'{metric}_precision'].append(
                    scores[metric].precision
                )
                all_scores[f'{metric}_recall'].append(
                    scores[metric].recall
                )
                all_scores[f'{metric}_fmeasure'].append(
                    scores[metric].fmeasure
                )

        return {
            k: np.mean(v)
            for k, v in all_scores.items()
        }

    def meteor_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        计算 METEOR 分数
        
        Returns:
            METEOR 分数 (0-1)
        """
        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk
            nltk.download('wordnet', quiet=True)

            scores = []
            for pred, ref in zip(predictions, references):
                score = meteor_score(
                    [ref.split()],
                    pred.split()
                )
                scores.append(score)

            return np.mean(scores)

        except Exception:
            return self._simple_meteor(predictions, references)

    def _simple_meteor(self, predictions: List[str], references: List[str]) -> float:
        """简化版 METEOR"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            # 简化的 unigram matching
            matches = 0
            for token in pred_tokens:
                if token in ref_tokens:
                    matches += 1
                    ref_tokens.remove(token)

            precision = matches / len(pred.split()) if pred else 0
            recall = matches / len(ref.split()) if ref else 0

            if precision + recall > 0:
                score = 2 * precision * recall / (precision + recall)
            else:
                score = 0

            scores.append(score)

        return np.mean(scores)

    def semantic_similarity(
        self,
        predictions: List[str],
        references: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> float:
        """
        计算语义相似度
        
        Returns:
            平均语义相似度 (0-1)
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            model = SentenceTransformer(model_name)

            pred_embeddings = model.encode(predictions)
            ref_embeddings = model.encode(references)

            similarities = cosine_similarity(pred_embeddings, ref_embeddings)
            diagonal = np.diag(similarities)

            return float(np.mean(diagonal))

        except ImportError:
            # 简化版：基于词重叠
            return self._simple_similarity(predictions, references)

    def _simple_similarity(self, predictions: List[str], references: List[str]) -> float:
        """简化版相似度"""
        similarities = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())

            if len(pred_words) == 0 and len(ref_words) == 0:
                similarities.append(1.0)
            elif len(pred_words) == 0 or len(ref_words) == 0:
                similarities.append(0.0)
            else:
                intersection = pred_words & ref_words
                union = pred_words | ref_words
                similarities.append(len(intersection) / len(union))

        return np.mean(similarities)

    def faithfulness_score(
        self,
        predictions: List[str],
        contexts: List[str]
    ) -> float:
        """
        计算忠实度分数
        
        基于预测中被上下文支持的事实比例
        """
        scores = []
        for pred, ctx in zip(predictions, contexts):
            pred_facts = self._extract_facts(pred)
            ctx_facts = self._extract_facts(ctx)

            if len(pred_facts) == 0:
                scores.append(1.0)  # 没有事实声称，视为忠实
            else:
                supported = len(pred_facts & ctx_facts)
                scores.append(supported / len(pred_facts))

        return np.mean(scores)

    def _extract_facts(self, text: str) -> set:
        """提取文本中的事实"""
        import re
        facts = set()

        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', text)
        facts.update(numbers)

        # 提取引号内容
        quotes = re.findall(r'"([^"]*)"', text)
        facts.update(quotes)

        # 提取关键名词短语（简化）
        words = text.split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if words[i][0].isupper() and words[i+1][0].isupper():
                facts.add(bigram)

        return facts
