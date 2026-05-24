"""
visualization.py — Biểu đồ và trực quan hóa cho Streamlit.

Module này cung cấp:
- Biểu đồ phân bố nhãn
- Confusion Matrix plot
- Feature Importance chart
- Biểu đồ so sánh mô hình
"""

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from typing import Optional

from src.config import CLASS_LABELS


# Bảng màu cho 4 nhãn
LABEL_COLORS = {
    "Tốt": "#10B981",
    "Khá": "#3B82F6",
    "Đạt": "#F59E0B",
    "Chưa đạt": "#EF4444",
}

COLOR_SEQUENCE = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444"]


def plot_label_distribution(df: pd.DataFrame, column: str = "learning_result_label") -> go.Figure:
    """Vẽ biểu đồ phân bố nhãn phân loại."""
    counts = df[column].value_counts().reindex(CLASS_LABELS, fill_value=0)

    fig = go.Figure(data=[
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[LABEL_COLORS.get(label, "#6B7280") for label in counts.index],
            text=counts.values,
            textposition="auto",
            textfont=dict(size=14, color="white"),
        )
    ])

    fig.update_layout(
        title="Phân bố kết quả học tập",
        xaxis_title="Mức phân loại",
        yaxis_title="Số lượng học sinh",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    return fig


def plot_label_pie(df: pd.DataFrame, column: str = "learning_result_label") -> go.Figure:
    """Vẽ biểu đồ tròn phân bố nhãn."""
    counts = df[column].value_counts().reindex(CLASS_LABELS, fill_value=0)

    fig = go.Figure(data=[
        go.Pie(
            labels=counts.index,
            values=counts.values,
            marker=dict(colors=[LABEL_COLORS.get(label, "#6B7280") for label in counts.index]),
            textinfo="label+percent+value",
            textfont=dict(size=13),
            hole=0.35,
        )
    ])

    fig.update_layout(
        title="Tỷ lệ phân loại kết quả học tập",
        template="plotly_white",
        height=400,
    )

    return fig


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Vẽ Confusion Matrix bằng heatmap."""
    cm_array = np.array(cm)

    # Tạo text annotations
    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=str(cm_array[i][j]),
                    font=dict(size=16, color="white" if cm_array[i][j] > cm_array.max() / 2 else "black"),
                    showarrow=False,
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=class_names,
        y=class_names,
        colorscale="Blues",
        showscale=True,
        text=cm_array,
        texttemplate="%{text}",
        textfont=dict(size=14),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Dự đoán",
        yaxis_title="Thực tế",
        yaxis=dict(autorange="reversed"),  # Đảo ngược trục y để đường chéo chính đi từ trái trên xuống phải dưới
        template="plotly_white",
        height=450,
        width=500,
    )

    return fig


def plot_feature_importance(
    importances: list[float],
    feature_names: list[str],
    top_n: int = 15,
    title: str = "Feature Importance — Random Forest",
) -> go.Figure:
    """Vẽ biểu đồ feature importance."""
    # Sort by importance
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    pairs = pairs[:top_n]
    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="#3B82F6",
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(300, len(names) * 35),
        yaxis=dict(autorange="reversed"),
    )

    return fig


def plot_metrics_comparison(
    dt_metrics: dict,
    rf_metrics: dict,
) -> go.Figure:
    """Vẽ biểu đồ so sánh metrics giữa 2 mô hình."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    dt_values = [
        dt_metrics["accuracy"],
        dt_metrics["precision_macro"],
        dt_metrics["recall_macro"],
        dt_metrics["f1_macro"],
    ]
    rf_values = [
        rf_metrics["accuracy"],
        rf_metrics["precision_macro"],
        rf_metrics["recall_macro"],
        rf_metrics["f1_macro"],
    ]

    fig = go.Figure(data=[
        go.Bar(
            name="Decision Tree",
            x=metrics_names,
            y=dt_values,
            marker_color="#F59E0B",
            text=[f"{v:.3f}" for v in dt_values],
            textposition="auto",
        ),
        go.Bar(
            name="Random Forest",
            x=metrics_names,
            y=rf_values,
            marker_color="#3B82F6",
            text=[f"{v:.3f}" for v in rf_values],
            textposition="auto",
        ),
    ])

    fig.update_layout(
        title="So sánh hiệu suất mô hình",
        yaxis_title="Giá trị",
        barmode="group",
        template="plotly_white",
        height=400,
        yaxis=dict(range=[0, 1.05]),
    )

    return fig


def plot_score_distribution(df: pd.DataFrame) -> go.Figure:
    """Vẽ histogram phân bố điểm trung bình."""
    fig = px.histogram(
        df,
        x="avg_score",
        color="learning_result_label",
        nbins=20,
        color_discrete_map=LABEL_COLORS,
        title="Phân bố điểm trung bình theo nhãn",
        labels={"avg_score": "Điểm trung bình", "learning_result_label": "Phân loại"},
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        barmode="overlay",
    )
    fig.update_traces(opacity=0.7)

    return fig
