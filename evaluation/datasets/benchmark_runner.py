"""
Benchmark 运行器
支持标准基准测试
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Benchmark 配置"""
    name: str
    dataset_path: str
    max_samples: int
    metrics: List[str]
    subset: Optional[str] = None
    split: str = "test"
    temperature: float = 0.7
    num_runs: int = 1


class BenchmarkRunner:
    """标准 Benchmark 运行器"""

    # 预定义的基准测试
    BENCHMARKS = {
        "mmlu": BenchmarkConfig(
            name="MMLU",
            dataset_path="cais/mmlu",
            max_samples=500,
            metrics=["accuracy", "f1_score"],
            subset="all"
        ),
        "hellaswag": BenchmarkConfig(
            name="HellaSwag",
            dataset_path="Rowan/hellaswag",
            max_samples=500,
            metrics=["accuracy"]
        ),
        "gsm8k": BenchmarkConfig(
            name="GSM8K",
            dataset_path="openai/gsm8k",
            max_samples=200,
            metrics=["accuracy", "f1_score"]
        ),
        "ceval": BenchmarkConfig(
            name="CEval",
            dataset_path="ceval/ceval-exam",
            max_samples=500,
            metrics=["accuracy", "f1_score"],
            subset="all"
        ),
        "cmmlu": BenchmarkConfig(
            name="CMMLU",
            dataset_path="haonan-li/cmmlu",
            max_samples=500,
            metrics=["accuracy", "f1_score"],
            subset="all"
        ),
        "humaneval": BenchmarkConfig(
            name="HumanEval",
            dataset_path="openai_humaneval",
            max_samples=164,
            metrics=["code_eval", "pass@1"]
        ),
        "mbpp": BenchmarkConfig(
            name="MBPP",
            dataset_path="google-research-datasets/mbpp",
            max_samples=200,
            metrics=["code_eval", "accuracy"]
        ),
    }

    def __init__(self, eval_engine):
        self.eval_engine = eval_engine
        self.results_cache = {}

    def get_available_benchmarks(self) -> List[str]:
        """获取可用的 benchmark 列表"""
        return list(self.BENCHMARKS.keys())

    def get_benchmark_info(self, name: str) -> Dict[str, Any]:
        """获取 benchmark 信息"""
        if name not in self.BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {name}")

        config = self.BENCHMARKS[name]
        return {
            "name": config.name,
            "dataset": config.dataset_path,
            "max_samples": config.max_samples,
            "metrics": config.metrics,
        }

    async def run_benchmark(
        self,
        benchmark_name: str,
        model_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行单个 benchmark
        
        Args:
            benchmark_name: benchmark 名称
            model_path: 模型路径
            **kwargs: 额外参数
        
        Returns:
            评估结果
        """
        if benchmark_name not in self.BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark: {benchmark_name}. "
                f"Available: {self.get_available_benchmarks()}"
            )

        config = self.BENCHMARKS[benchmark_name]
        logger.info(f"Running benchmark: {config.name}")

        # 加载数据集
        dataset = await self._load_benchmark_dataset(config)

        # 运行评估
        results = await self.eval_engine.run_evaluation(
            model_path=model_path,
            dataset_name=config.name,
            metrics=config.metrics,
            dataset=dataset,
            **kwargs
        )

        # 缓存结果
        self.results_cache[f"{model_path}:{benchmark_name}"] = results

        return results

    async def run_all_benchmarks(
        self,
        model_path: str,
        benchmarks: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行所有 benchmark
        
        Args:
            model_path: 模型路径
            benchmarks: 要运行的 benchmark 列表，None 表示全部
            **kwargs: 额外参数
        
        Returns:
            所有 benchmark 的结果
        """
        if benchmarks is None:
            benchmarks = self.get_available_benchmarks()

        all_results = {}
        for benchmark_name in benchmarks:
            try:
                results = await self.run_benchmark(
                    benchmark_name, model_path, **kwargs
                )
                all_results[benchmark_name] = results
            except Exception as e:
                logger.error(f"Benchmark {benchmark_name} failed: {e}")
                all_results[benchmark_name] = {"error": str(e)}

        return all_results

    async def _load_benchmark_dataset(
        self,
        config: BenchmarkConfig
    ) -> List[Dict]:
        """加载 benchmark 数据集"""
        try:
            from datasets import load_dataset

            # 加载数据集
            if config.subset:
                dataset = load_dataset(
                    config.dataset_path,
                    config.subset,
                    split=config.split,
                    streaming=True
                )
            else:
                dataset = load_dataset(
                    config.dataset_path,
                    split=config.split,
                    streaming=True
                )

            # 转换为列表
            samples = []
            for i, sample in enumerate(dataset):
                if i >= config.max_samples:
                    break
                samples.append(self._process_sample(sample, config.name))

            return samples

        except Exception as e:
            logger.error(f"Failed to load benchmark dataset: {e}")
            raise

    def _process_sample(self, sample: Dict, benchmark_name: str) -> Dict:
        """处理单个样本"""
        if benchmark_name == "mmlu" or benchmark_name == "cmmlu":
            # MMLU 格式
            question = sample.get("question", "")
            choices = sample.get("choices", {})
            answer = sample.get("answer", 0)

            prompt = f"{question}\n\n"
            prompt += "A. " + choices.get("A", "") + "\n"
            prompt += "B. " + choices.get("B", "") + "\n"
            prompt += "C. " + choices.get("C", "") + "\n"
            prompt += "D. " + choices.get("D", "") + "\n"
            prompt += "Please answer with the letter (A, B, C, or D)."

            return {
                "prompt": prompt,
                "reference": chr(65 + answer),  # A, B, C, D
                "metadata": {"subject": sample.get("subject", "")}
            }

        elif benchmark_name == "ceval":
            # CEval 格式
            question = sample.get("question", "")
            choice_a = sample.get("A", "")
            choice_b = sample.get("B", "")
            choice_c = sample.get("C", "")
            choice_d = sample.get("D", "")
            answer = sample.get("answer", "A")

            prompt = f"{question}\n\n"
            prompt += f"A. {choice_a}\n"
            prompt += f"B. {choice_b}\n"
            prompt += f"C. {choice_c}\n"
            prompt += f"D. {choice_d}\n"
            prompt += "Please answer with the letter (A, B, C, or D)."

            return {
                "prompt": prompt,
                "reference": answer,
                "metadata": {"subject": sample.get("subject", "")}
            }

        elif benchmark_name == "gsm8k":
            # GSM8K 格式
            question = sample.get("question", "")
            answer = sample.get("answer", "")

            return {
                "prompt": f"Question: {question}\n\nLet's solve this step by step.",
                "reference": answer,
                "metadata": {}
            }

        elif benchmark_name == "hellaswag":
            # HellaSwag 格式
            ctx = sample.get("ctx", "")
            endings = sample.get("endings", [])
            label = sample.get("label", 0)

            prompt = f"{ctx}\n"
            for i, ending in enumerate(endings):
                prompt += f"{chr(65+i)}. {ending}\n"
            prompt += "Which continuation is correct? (A, B, C, or D)"

            return {
                "prompt": prompt,
                "reference": chr(65 + label),
                "metadata": {}
            }

        else:
            # 通用格式
            return {
                "prompt": sample.get("question", sample.get("prompt", "")),
                "reference": sample.get("answer", sample.get("reference", "")),
                "metadata": {}
            }

    def compare_benchmarks(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """
        比较两个模型的 benchmark 结果
        """
        comparison = {
            "model1": model1_name,
            "model2": model2_name,
            "benchmarks": {}
        }

        common_benchmarks = set(results1.keys()) & set(results2.keys())

        for benchmark in common_benchmarks:
            if "error" in str(results1.get(benchmark, {})) or \
               "error" in str(results2.get(benchmark, {})):
                continue

            # 比较指标
            r1 = results1[benchmark]
            r2 = results2[benchmark]

            if hasattr(r1, 'metrics') and hasattr(r2, 'metrics'):
                metrics_comparison = {}
                for metric in set(r1.metrics.keys()) & set(r2.metrics.keys()):
                    val1 = r1.metrics[metric].value
                    val2 = r2.metrics[metric].value

                    metrics_comparison[metric] = {
                        model1_name: val1,
                        model2_name: val2,
                        "difference": val2 - val1,
                        "improvement": val2 > val1
                    }

                comparison["benchmarks"][benchmark] = metrics_comparison

        return comparison

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ):
        """导出结果"""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # 转换为可序列化格式
            serializable = {}
            for key, value in results.items():
                if hasattr(value, '__dict__'):
                    serializable[key] = value.__dict__
                else:
                    serializable[key] = value

            with open(output, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Results exported to {output}")
