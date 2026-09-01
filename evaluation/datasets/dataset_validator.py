"""
数据集验证器
检查数据集的质量和一致性
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class DatasetValidator:
    """数据集验证器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_samples = self.config.get("min_samples", 10)
        self.max_length = self.config.get("max_length", 4096)
        self.required_fields = self.config.get("required_fields", ["prompt"])

    def validate(self, dataset: List[Dict]) -> ValidationResult:
        """
        验证数据集
        
        Args:
            dataset: 数据集列表
        
        Returns:
            验证结果
        """
        result = ValidationResult(is_valid=True)

        # 检查基本条件
        if len(dataset) == 0:
            result.is_valid = False
            result.errors.append("Dataset is empty")
            return result

        if len(dataset) < self.min_samples:
            result.warnings.append(
                f"Dataset has only {len(dataset)} samples, "
                f"recommended at least {self.min_samples}"
            )

        # 检查字段
        missing_fields = self._check_fields(dataset)
        if missing_fields:
            result.is_valid = False
            result.errors.append(f"Missing required fields: {missing_fields}")

        # 检查空值
        null_issues = self._check_nulls(dataset)
        if null_issues:
            result.warnings.extend(null_issues)

        # 检查重复
        duplicates = self._check_duplicates(dataset)
        if duplicates > 0:
            result.warnings.append(f"Found {duplicates} duplicate samples")

        # 检查长度
        length_issues = self._check_lengths(dataset)
        if length_issues:
            result.warnings.extend(length_issues)

        # 计算统计信息
        result.stats = self._compute_stats(dataset)

        return result

    def _check_fields(self, dataset: List[Dict]) -> List[str]:
        """检查必需字段"""
        missing = []

        for i, item in enumerate(dataset[:10]):  # 只检查前10个
            for field in self.required_fields:
                if field not in item:
                    missing.append(f"Sample {i}: missing '{field}'")

        return missing

    def _check_nulls(self, dataset: List[Dict]) -> List[str]:
        """检查空值"""
        issues = []

        for i, item in enumerate(dataset):
            for key, value in item.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    issues.append(f"Sample {i}: '{key}' is empty")

        return issues[:10]  # 最多返回10个

    def _check_duplicates(self, dataset: List[Dict]) -> int:
        """检查重复"""
        seen = set()
        duplicates = 0

        for item in dataset:
            key = str(item)
            if key in seen:
                duplicates += 1
            seen.add(key)

        return duplicates

    def _check_lengths(self, dataset: List[Dict]) -> List[str]:
        """检查长度"""
        issues = []
        too_long = 0

        for i, item in enumerate(dataset):
            prompt = item.get("prompt", "")
            if len(prompt) > self.max_length:
                too_long += 1

        if too_long > 0:
            issues.append(
                f"{too_long} samples exceed max length of {self.max_length}"
            )

        return issues

    def _compute_stats(self, dataset: List[Dict]) -> Dict[str, Any]:
        """计算统计信息"""
        stats = {
            "total_samples": len(dataset),
            "fields": list(dataset[0].keys()) if dataset else [],
        }

        # 计算长度分布
        lengths = []
        for item in dataset:
            prompt = item.get("prompt", "")
            lengths.append(len(prompt))

        if lengths:
            import numpy as np
            stats["length_stats"] = {
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "std": float(np.std(lengths)),
            }

        return stats
