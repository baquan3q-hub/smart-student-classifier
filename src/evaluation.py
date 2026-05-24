"""
evaluation.py — Đánh giá mô hình Machine Learning.

Module này cung cấp:
- Tính metrics (Accuracy, Precision, Recall, F1-score)
- Confusion Matrix
- Classification Report
- Feature Importance
"""

import numpy as np
import pandas as pd
from typing import Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Đánh giá mô hình bằng các metrics tiêu chuẩn.

    Args:
        model: Mô hình đã train.
        X_test: Features test set.
        y_test: Labels test set (encoded).
        class_names: Tên các lớp.
        model_name: Tên mô hình cho báo cáo.

    Returns:
        Dictionary chứa tất cả metrics.
    """
    y_pred = model.predict(X_test)
    
    # Định nghĩa thứ tự nhãn yêu cầu
    target_labels = ["Tốt", "Khá", "Đạt", "Chưa đạt"]
    # Ánh xạ tên nhãn sang số nguyên tương ứng dựa trên class_names (le.classes_)
    labels_encoded = [list(class_names).index(label) for label in target_labels]

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels_encoded).tolist(),
        "class_names": target_labels,
    }

    # Feature importance (nếu model hỗ trợ)
    if hasattr(model, "feature_importances_"):
        metrics["feature_importances"] = model.feature_importances_.tolist()

    return metrics


def generate_classification_report(
    dt_model,
    rf_model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: np.ndarray,
) -> str:
    """
    Sinh báo cáo classification report dạng text.

    Args:
        dt_model: Decision Tree model.
        rf_model: Random Forest model.
        X_test: Features test set.
        y_test: Labels test set.
        class_names: Tên các lớp.

    Returns:
        Chuỗi text chứa báo cáo đầy đủ.
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("CLASSIFICATION REPORT - Smart Student Classification System")
    report_lines.append("=" * 60)

    for model, name in [(dt_model, "Decision Tree"), (rf_model, "Random Forest")]:
        y_pred = model.predict(X_test)
        report_lines.append(f"\n--- {name} ---\n")
        report_lines.append(
            classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        )

    return "\n".join(report_lines)


def get_feature_importance_df(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Trả về DataFrame feature importance sắp xếp giảm dần.

    Args:
        model: Mô hình có thuộc tính feature_importances_.
        feature_names: Tên các features.

    Returns:
        DataFrame với cột 'feature' và 'importance'.
    """
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["importance"] = importance_df["importance"].round(4)

    return importance_df


def compare_models(dt_metrics: dict, rf_metrics: dict) -> pd.DataFrame:
    """
    So sánh 2 mô hình bằng bảng metrics.

    Args:
        dt_metrics: Metrics của Decision Tree.
        rf_metrics: Metrics của Random Forest.

    Returns:
        DataFrame so sánh.
    """
    comparison = pd.DataFrame({
        "Chỉ số": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"],
        "Decision Tree": [
            dt_metrics["accuracy"],
            dt_metrics["precision_macro"],
            dt_metrics["recall_macro"],
            dt_metrics["f1_macro"],
        ],
        "Random Forest": [
            rf_metrics["accuracy"],
            rf_metrics["precision_macro"],
            rf_metrics["recall_macro"],
            rf_metrics["f1_macro"],
        ],
    })

    # Highlight winner
    comparison["Mô hình tốt hơn"] = comparison.apply(
        lambda row: "Random Forest" if row["Random Forest"] >= row["Decision Tree"] else "Decision Tree",
        axis=1,
    )

    return comparison
