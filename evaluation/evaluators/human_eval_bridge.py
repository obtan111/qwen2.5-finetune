"""
人工评估桥接
支持人工标注和评估结果整合
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class HumanEvalBridge:
    """人工评估桥接"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化人工评估桥接
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.output_dir = Path(self.config.get("output_dir", "./human_eval"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_evaluation_data(
        self,
        predictions: List[Dict],
        output_file: str = "eval_data.jsonl"
    ) -> str:
        """
        准备人工评估数据
        
        Args:
            predictions: 预测结果列表
            output_file: 输出文件名
        
        Returns:
            输出文件路径
        """
        output_path = self.output_dir / output_file

        eval_data = []
        for i, pred in enumerate(predictions):
            eval_data.append({
                "id": i,
                "prompt": pred.get("prompt", ""),
                "prediction": pred.get("prediction", ""),
                "reference": pred.get("reference", ""),
                "metadata": pred.get("metadata", {}),
                "human_scores": {}  # 待人工填写
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in eval_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Prepared {len(eval_data)} samples for human evaluation")
        return str(output_path)

    def load_human_scores(self, scores_file: str) -> List[Dict]:
        """
        加载人工评分结果
        
        Args:
            scores_file: 评分文件路径
        
        Returns:
            评分结果列表
        """
        scores = []
        with open(scores_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    scores.append(json.loads(line))

        return scores

    def merge_scores(
        self,
        predictions: List[Dict],
        human_scores: List[Dict]
    ) -> List[Dict]:
        """
        合并预测结果和人工评分
        
        Args:
            predictions: 预测结果列表
            human_scores: 人工评分列表
        
        Returns:
            合并后的结果列表
        """
        # 按 ID 索引人工评分
        scores_map = {s["id"]: s for s in human_scores}

        merged = []
        for pred in predictions:
            item_id = pred.get("id", 0)
            human_score = scores_map.get(item_id, {})

            merged_item = {
                **pred,
                "human_scores": human_score.get("human_scores", {}),
                "human_annotation": human_score.get("annotation", "")
            }
            merged.append(merged_item)

        return merged

    def compute_inter_annotator_agreement(
        self,
        annotator_scores: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        计算标注者间一致性
        
        Args:
            annotator_scores: {annotator_id: [scores]} 格式
        
        Returns:
            一致性指标
        """
        import numpy as np
        from scipy import stats

        # 收集所有标注者的分数
        annotators = list(annotator_scores.keys())
        if len(annotators) < 2:
            return {"error": "Need at least 2 annotators"}

        # 计算 Cohen's Kappa
        all_scores = []
        for annotator in annotators:
            scores = [s.get("score", 3) for s in annotator_scores[annotator]]
            all_scores.append(scores)

        # 两两计算 Kappa
        kappa_scores = []
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                kappa = self._compute_kappa(all_scores[i], all_scores[j])
                kappa_scores.append({
                    "annotator_pair": (annotators[i], annotators[j]),
                    "kappa": kappa
                })

        # 计算平均一致性
        avg_kappa = np.mean([k["kappa"] for k in kappa_scores])

        return {
            "num_annotators": len(annotators),
            "pairwise_kappa": kappa_scores,
            "average_kappa": float(avg_kappa),
            "agreement_level": self._interpret_kappa(avg_kappa)
        }

    def _compute_kappa(self, scores1: List[int], scores2: List[int]) -> float:
        """计算 Cohen's Kappa"""
        import numpy as np

        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have same length")

        n = len(scores1)
        categories = set(scores1 + scores2)

        # 计算观测一致率
        agreements = sum(1 for s1, s2 in zip(scores1, scores2) if s1 == s2)
        po = agreements / n

        # 计算期望一致率
        pe = 0
        for cat in categories:
            p1 = sum(1 for s in scores1 if s == cat) / n
            p2 = sum(1 for s in scores2 if s == cat) / n
            pe += p1 * p2

        # 计算 Kappa
        if pe == 1:
            return 1.0

        kappa = (po - pe) / (1 - pe)
        return float(kappa)

    def _interpret_kappa(self, kappa: float) -> str:
        """解释 Kappa 值"""
        if kappa < 0:
            return "Poor"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"

    def generate_annotation_template(
        self,
        criteria: List[str],
        output_file: str = "annotation_template.json"
    ) -> str:
        """
        生成标注模板
        
        Args:
            criteria: 评估标准列表
            output_file: 输出文件名
        
        Returns:
            输出文件路径
        """
        template = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "criteria": [],
            "instructions": "请根据以下标准对每条响应进行评分（1-5分）"
        }

        for criterion in criteria:
            template["criteria"].append({
                "name": criterion,
                "description": self._get_criterion_description(criterion),
                "scale": {
                    1: "非常差",
                    2: "差",
                    3: "一般",
                    4: "好",
                    5: "非常好"
                }
            })

        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def _get_criterion_description(self, criterion: str) -> str:
        """获取标准描述"""
        descriptions = {
            "relevance": "答案是否与问题相关",
            "coherence": "答案是否逻辑连贯",
            "factuality": "答案是否基于事实",
            "helpfulness": "答案是否对用户有帮助",
            "safety": "答案是否安全无害",
            "fluency": "语言是否流畅自然",
            "completeness": "答案是否完整"
        }
        return descriptions.get(criterion, f"评估 {criterion}")
