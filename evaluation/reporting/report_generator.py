"""
报告生成器
生成完整的评估报告
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get("dir", "./results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = config.get("format", ["json", "html"])

    def generate(
        self,
        model_name: str,
        dataset_name: str,
        metrics: Dict[str, Any],
        predictions: Optional[List[Dict]] = None,
        resource_usage: Optional[Dict] = None,
        comparison_with_baseline: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        生成完整评估报告
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            metrics: 指标结果
            predictions: 预测结果
            resource_usage: 资源使用情况
            comparison_with_baseline: 与基线的比较
        
        Returns:
            报告数据
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "metadata": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "timestamp": datetime.now().isoformat(),
                "report_id": f"{model_name}_{dataset_name}_{timestamp}"
            },
            "metrics": self._format_metrics(metrics),
            "summary": self._generate_summary(metrics),
            "resource_usage": resource_usage,
            "comparison": comparison_with_baseline,
        }

        # 保存报告
        self._save_report(report, timestamp)

        # 保存预测结果（如果需要）
        if predictions and self.config.get("save_predictions", True):
            self._save_predictions(predictions, timestamp)

        # 生成可视化（如果配置了）
        if "html" in self.formats:
            self._generate_html_report(report, timestamp)

        return report

    def _format_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """格式化指标"""
        formatted = {}

        for name, value in metrics.items():
            if hasattr(value, 'value'):
                formatted[name] = {
                    "value": value.value,
                    "confidence_interval": value.confidence_interval,
                    "sample_size": value.sample_size,
                    "computation_time": value.computation_time,
                    "metadata": value.metadata
                }
            elif isinstance(value, dict):
                formatted[name] = value
            else:
                formatted[name] = {"value": value}

        return formatted

    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要"""
        summary = {
            "total_metrics": len(metrics),
            "successful_metrics": 0,
            "failed_metrics": 0,
            "key_findings": []
        }

        for name, value in metrics.items():
            if hasattr(value, 'value'):
                if not (value.value != value.value):  # Check for NaN
                    summary["successful_metrics"] += 1
                else:
                    summary["failed_metrics"] += 1
            else:
                summary["successful_metrics"] += 1

        # 关键发现
        if "accuracy" in metrics:
            acc = metrics["accuracy"]
            if hasattr(acc, 'value'):
                summary["key_findings"].append(
                    f"Accuracy: {acc.value:.4f}"
                )

        if "f1_score" in metrics:
            f1 = metrics["f1_score"]
            if hasattr(f1, 'value'):
                summary["key_findings"].append(
                    f"F1 Score: {f1.value:.4f}"
                )

        return summary

    def _save_report(self, report: Dict, timestamp: str):
        """保存报告"""
        if "json" in self.formats:
            output_path = self.output_dir / f"report_{timestamp}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Report saved to {output_path}")

    def _save_predictions(self, predictions: List[Dict], timestamp: str):
        """保存预测结果"""
        output_path = self.output_dir / f"predictions_{timestamp}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        logger.info(f"Predictions saved to {output_path}")

    def _generate_html_report(self, report: Dict, timestamp: str):
        """生成 HTML 报告"""
        html_content = self._render_html(report)
        output_path = self.output_dir / f"report_{timestamp}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML report saved to {output_path}")

    def _render_html(self, report: Dict) -> str:
        """渲染 HTML 报告"""
        metadata = report["metadata"]
        metrics = report["metrics"]
        summary = report["summary"]

        metrics_rows = ""
        for name, value in metrics.items():
            if isinstance(value, dict) and "value" in value:
                val = value["value"]
                ci = value.get("confidence_interval")
                ci_text = f"({ci[0]:.4f}, {ci[1]:.4f})" if ci else "N/A"
                metrics_rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>{val:.4f}</td>
                    <td>{ci_text}</td>
                </tr>
                """

        findings = "\n".join([
            f"<li>{f}</li>" for f in summary.get("key_findings", [])
        ])

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>评估报告 - {metadata['model_name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
        .metric-value {{ font-weight: bold; color: #2196F3; }}
    </style>
</head>
<body>
    <h1>模型评估报告</h1>
    
    <div class="summary">
        <h2>元数据</h2>
        <p><strong>模型:</strong> {metadata['model_name']}</p>
        <p><strong>数据集:</strong> {metadata['dataset_name']}</p>
        <p><strong>时间:</strong> {metadata['timestamp']}</p>
    </div>
    
    <h2>评估指标</h2>
    <table>
        <thead>
            <tr>
                <th>指标</th>
                <th>值</th>
                <th>95% 置信区间</th>
            </tr>
        </thead>
        <tbody>
            {metrics_rows}
        </tbody>
    </table>
    
    <div class="summary">
        <h2>摘要</h2>
        <p>成功计算: {summary['successful_metrics']} / {summary['total_metrics']}</p>
        <h3>关键发现:</h3>
        <ul>{findings}</ul>
    </div>
</body>
</html>"""

        return html

    def load_report(self, report_path: str) -> Dict[str, Any]:
        """加载报告"""
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compare_reports(
        self,
        report1: Dict[str, Any],
        report2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """比较两个报告"""
        comparison = {
            "report1": report1["metadata"],
            "report2": report2["metadata"],
            "metrics_comparison": {}
        }

        metrics1 = report1.get("metrics", {})
        metrics2 = report2.get("metrics", {})

        common_metrics = set(metrics1.keys()) & set(metrics2.keys())

        for metric in common_metrics:
            val1 = metrics1[metric].get("value", 0)
            val2 = metrics2[metric].get("value", 0)

            comparison["metrics_comparison"][metric] = {
                "report1_value": val1,
                "report2_value": val2,
                "difference": val2 - val1,
                "relative_change": (val2 - val1) / val1 if val1 != 0 else float('inf'),
                "improvement": val2 > val1
            }

        return comparison
