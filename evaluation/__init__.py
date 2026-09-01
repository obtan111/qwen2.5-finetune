"""
评估框架主入口
"""

from .core import EvalEngine, ModelManager, CacheManager, ResourceMonitor
from .metrics import MetricsCalculator, BaseMetrics, GenerationMetrics, TaskMetrics, StatisticalMetrics
from .datasets import DatasetLoader, DatasetValidator, BenchmarkRunner
from .evaluators import LLMJudge, SafetyEvaluator, HumanEvalBridge
from .reporting import ReportGenerator, EvalVisualizer, ExportUtils

__version__ = "1.0.0"
__author = "Enterprise AI Evaluation Team"

__all__ = [
    # Core
    "EvalEngine",
    "ModelManager",
    "CacheManager",
    "ResourceMonitor",
    
    # Metrics
    "MetricsCalculator",
    "BaseMetrics",
    "GenerationMetrics",
    "TaskMetrics",
    "StatisticalMetrics",
    
    # Datasets
    "DatasetLoader",
    "DatasetValidator",
    "BenchmarkRunner",
    
    # Evaluators
    "LLMJudge",
    "SafetyEvaluator",
    "HumanEvalBridge",
    
    # Reporting
    "ReportGenerator",
    "EvalVisualizer",
    "ExportUtils",
]
