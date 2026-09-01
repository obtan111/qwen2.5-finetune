"""
可视化模块
生成各种评估图表
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EvalVisualizer:
    """评估可视化器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(
        self,
        train_losses: List[float],
        eval_losses: List[float],
        learning_rates: Optional[List[float]] = None,
        save_path: str = "training_curves.html"
    ):
        """绘制训练曲线"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            if learning_rates:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "Training Loss",
                        "Evaluation Loss",
                        "Learning Rate",
                        "Loss Comparison"
                    )
                )
            else:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Training Loss", "Evaluation Loss")
                )

            # Training Loss
            fig.add_trace(
                go.Scatter(
                    y=train_losses,
                    name="Train Loss",
                    line=dict(color="blue")
                ),
                row=1, col=1
            )

            # Evaluation Loss
            fig.add_trace(
                go.Scatter(
                    y=eval_losses,
                    name="Eval Loss",
                    line=dict(color="red")
                ),
                row=1, col=2
            )

            if learning_rates:
                # Learning Rate
                fig.add_trace(
                    go.Scatter(
                        y=learning_rates,
                        name="Learning Rate",
                        line=dict(color="green")
                    ),
                    row=2, col=1
                )

                # Loss Comparison
                fig.add_trace(
                    go.Scatter(y=train_losses, name="Train", line=dict(color="blue")),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(y=eval_losses, name="Eval", line=dict(color="red")),
                    row=2, col=2
                )

            fig.update_layout(
                height=800,
                width=1200,
                title_text="Training Progress"
            )

            output_path = self.output_dir / save_path
            fig.write_html(str(output_path))
            logger.info(f"Training curves saved to {output_path}")

        except ImportError:
            logger.warning("plotly not installed, skipping visualization")

    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: str = "metrics_comparison.html"
    ):
        """绘制指标对比图"""
        try:
            import plotly.graph_objects as go
            import pandas as pd

            df = pd.DataFrame(results).T

            fig = go.Figure()

            for column in df.columns:
                fig.add_trace(go.Bar(
                    name=column,
                    x=df.index,
                    y=df[column]
                ))

            fig.update_layout(
                barmode='group',
                title="Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                height=600,
                width=1000
            )

            output_path = self.output_dir / save_path
            fig.write_html(str(output_path))
            logger.info(f"Metrics comparison saved to {output_path}")

        except ImportError:
            logger.warning("plotly not installed, skipping visualization")

    def plot_radar_chart(
        self,
        metrics: Dict[str, float],
        title: str = "Model Performance Radar",
        save_path: str = "radar_chart.html"
    ):
        """绘制雷达图"""
        try:
            import plotly.graph_objects as go

            categories = list(metrics.keys())
            values = list(metrics.values())
            values.append(values[0])  # 闭合

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name='Model Performance'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title=title,
                height=600,
                width=600
            )

            output_path = self.output_dir / save_path
            fig.write_html(str(output_path))
            logger.info(f"Radar chart saved to {output_path}")

        except ImportError:
            logger.warning("plotly not installed, skipping visualization")

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: List[str],
        save_path: str = "confusion_matrix.html"
    ):
        """绘制混淆矩阵"""
        try:
            import plotly.graph_objects as go
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)

            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Blues'
            ))

            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='True',
                height=600,
                width=600
            )

            output_path = self.output_dir / save_path
            fig.write_html(str(output_path))
            logger.info(f"Confusion matrix saved to {output_path}")

        except ImportError:
            logger.warning("plotly/scikit-learn not installed, skipping visualization")

    def plot_distribution(
        self,
        values: List[float],
        title: str = "Score Distribution",
        save_path: str = "distribution.html"
    ):
        """绘制分布图"""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=values,
                nbinsx=30,
                name="Distribution"
            ))

            # 添加均值线
            mean_val = np.mean(values)
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.4f}"
            )

            fig.update_layout(
                title=title,
                xaxis_title="Score",
                yaxis_title="Count",
                height=500,
                width=800
            )

            output_path = self.output_dir / save_path
            fig.write_html(str(output_path))
            logger.info(f"Distribution plot saved to {output_path}")

        except ImportError:
            logger.warning("plotly not installed, skipping visualization")

    def plot_box_plot(
        self,
        data: Dict[str, List[float]],
        title: str = "Score Box Plot",
        save_path: str = "box_plot.html"
    ):
        """绘制箱线图"""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            for name, values in data.items():
                fig.add_trace(go.Box(
                    y=values,
                    name=name
                ))

            fig.update_layout(
                title=title,
                yaxis_title="Score",
                height=500,
                width=800
            )

            output_path = self.output_dir / save_path
            fig.write_html(str(output_path))
            logger.info(f"Box plot saved to {output_path}")

        except ImportError:
            logger.warning("plotly not installed, skipping visualization")
