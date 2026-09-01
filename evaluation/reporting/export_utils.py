"""
导出工具
支持多种格式导出
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ExportUtils:
    """导出工具"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        data: Dict[str, Any],
        filename: str,
        indent: int = 2
    ) -> str:
        """
        导出 JSON 文件
        
        Args:
            data: 数据
            filename: 文件名
            indent: 缩进
        
        Returns:
            文件路径
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

        logger.info(f"Exported JSON to {output_path}")
        return str(output_path)

    def export_csv(
        self,
        data: List[Dict[str, Any]],
        filename: str
    ) -> str:
        """
        导出 CSV 文件
        
        Args:
            data: 数据列表
            filename: 文件名
        
        Returns:
            文件路径
        """
        if not data:
            logger.warning("No data to export")
            return ""

        output_path = self.output_dir / filename

        # 获取所有字段
        fieldnames = list(data[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"Exported CSV to {output_path}")
        return str(output_path)

    def export_jsonl(
        self,
        data: List[Dict[str, Any]],
        filename: str
    ) -> str:
        """
        导出 JSONL 文件
        
        Args:
            data: 数据列表
            filename: 文件名
        
        Returns:
            文件路径
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Exported JSONL to {output_path}")
        return str(output_path)

    def export_metrics_table(
        self,
        metrics: Dict[str, Any],
        filename: str
    ) -> str:
        """
        导出指标表格
        """
        rows = []
        for name, value in metrics.items():
            if isinstance(value, dict) and "value" in value:
                rows.append({
                    "metric": name,
                    "value": value["value"],
                    "confidence_interval": str(value.get("confidence_interval", "N/A")),
                    "sample_size": value.get("sample_size", "N/A")
                })
            else:
                rows.append({
                    "metric": name,
                    "value": value,
                    "confidence_interval": "N/A",
                    "sample_size": "N/A"
                })

        return self.export_csv(rows, filename)

    def export_comparison(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        name1: str = "Model 1",
        name2: str = "Model 2",
        filename: str = "comparison.csv"
    ) -> str:
        """
        导出比较结果
        """
        rows = []
        common_metrics = set(results1.keys()) & set(results2.keys())

        for metric in sorted(common_metrics):
            val1 = results1[metric]
            val2 = results2[metric]

            if isinstance(val1, dict) and "value" in val1:
                val1 = val1["value"]
            if isinstance(val2, dict) and "value" in val2:
                val2 = val2["value"]

            try:
                diff = float(val2) - float(val1)
                pct_change = (diff / float(val1) * 100) if float(val1) != 0 else 0
            except:
                diff = "N/A"
                pct_change = "N/A"

            rows.append({
                "metric": metric,
                name1: val1,
                name2: val2,
                "difference": diff,
                "percent_change": f"{pct_change:.2f}%" if isinstance(pct_change, float) else pct_change,
                "improved": "Yes" if isinstance(diff, float) and diff > 0 else "No"
            })

        return self.export_csv(rows, filename)

    def export_report_html(
        self,
        report: Dict[str, Any],
        filename: str = "report.html"
    ) -> str:
        """
        导出 HTML 报告
        """
        from .report_generator import ReportGenerator
        generator = ReportGenerator({"dir": str(self.output_dir)})
        return generator._generate_html_report(report, filename.replace(".html", ""))

    def list_files(self) -> List[str]:
        """列出导出目录中的文件"""
        return [
            str(f.name)
            for f in self.output_dir.iterdir()
            if f.is_file()
        ]

    def cleanup_old_files(self, max_age_days: int = 30):
        """清理旧文件"""
        import time
        current_time = time.time()

        for file_path in self.output_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_days * 86400:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
