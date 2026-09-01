"""
统计指标
置信区间、显著性检验、分布分析等
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import stats


class StatisticalMetrics:
    """统计指标"""

    @staticmethod
    def confidence_interval(
        values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        计算置信区间
        
        Args:
            values: 数值列表
            confidence: 置信水平 (0-1)
        
        Returns:
            (下界, 上界) 元组
        """
        if len(values) == 0:
            return (0.0, 0.0)

        values = np.array(values)
        n = len(values)

        if n == 1:
            return (values[0], values[0])

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / np.sqrt(n)

        # t 分布
        t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)

        ci_lower = mean - t_value * se
        ci_upper = mean + t_value * se

        return (float(ci_lower), float(ci_upper))

    @staticmethod
    def bootstrap_confidence_interval(
        values: List[float],
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Bootstrap 置信区间
        """
        if len(values) == 0:
            return (0.0, 0.0)

        values = np.array(values)
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)
        lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

        return (float(lower), float(upper))

    @staticmethod
    def paired_t_test(
        values1: List[float],
        values2: List[float],
        significance: float = 0.05
    ) -> dict:
        """
        配对 t 检验
        
        用于比较两个模型在相同数据集上的表现差异
        """
        if len(values1) != len(values2):
            raise ValueError("Sample sizes must be equal")

        values1 = np.array(values1)
        values2 = np.array(values2)

        t_stat, p_value = stats.ttest_rel(values1, values2)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < significance,
            "mean_diff": float(np.mean(values2 - values1)),
            "ci": StatisticalMetrics.confidence_interval(
                (values2 - values1).tolist()
            )
        }

    @staticmethod
    def mann_whitney_u_test(
        values1: List[float],
        values2: List[float],
        significance: float = 0.05
    ) -> dict:
        """
        Mann-Whitney U 检验
        非参数检验，不需要正态分布假设
        """
        values1 = np.array(values1)
        values2 = np.array(values2)

        u_stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')

        return {
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant": p_value < significance,
            "effect_size": float(1 - 2 * u_stat / (len(values1) * len(values2)))
        }

    @staticmethod
    def cohens_d(values1: List[float], values2: List[float]) -> float:
        """
        计算 Cohen's d 效应量
        
        返回值解释：
        - 0.2: 小效应
        - 0.5: 中效应
        - 0.8: 大效应
        """
        values1 = np.array(values1)
        values2 = np.array(values2)

        n1, n2 = len(values1), len(values2)
        var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)

        # 合并标准差
        pooled_std = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        )

        if pooled_std == 0:
            return 0.0

        d = (np.mean(values2) - np.mean(values1)) / pooled_std

        return float(d)

    @staticmethod
    def distribution_analysis(values: List[float]) -> dict:
        """
        分布分析
        """
        values = np.array(values)

        if len(values) == 0:
            return {}

        # 基本统计
        result = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "skewness": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values)),
            "percentiles": {
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }
        }

        # 正态性检验
        if len(values) >= 8:
            stat, p_value = stats.shapiro(values[:min(5000, len(values))])
            result["normality_test"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05
            }

        return result

    @staticmethod
    def correlation_analysis(
        values1: List[float],
        values2: List[float]
    ) -> dict:
        """
        相关性分析
        """
        values1 = np.array(values1)
        values2 = np.array(values2)

        # Pearson 相关
        pearson_r, pearson_p = stats.pearsonr(values1, values2)

        # Spearman 相关
        spearman_r, spearman_p = stats.spearmanr(values1, values2)

        return {
            "pearson": {
                "correlation": float(pearson_r),
                "p_value": float(pearson_p),
                "significant": pearson_p < 0.05
            },
            "spearman": {
                "correlation": float(spearman_r),
                "p_value": float(spearman_p),
                "significant": spearman_p < 0.05
            }
        }

    @staticmethod
    def effect_size_interpretation(d: float) -> str:
        """解释效应量"""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
