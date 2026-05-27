"""
predict.py — Dự đoán kết quả học tập học sinh.

Module này cung cấp:
- Dự đoán một học sinh (predict_one)
- Dự đoán hàng loạt (predict_batch)
- Load models đã train
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

from src.config import (
    FEATURE_COLUMNS, CLASS_LABELS,
    DECISION_TREE_MODEL_FILE, RANDOM_FOREST_MODEL_FILE, LABEL_ENCODER_FILE,
)
from src.rule_engine import classify_learning_result, get_classification_reason


def load_trained_models() -> tuple:
    """
    Load các mô hình đã huấn luyện.

    Returns:
        (dt_model, rf_model, label_encoder)

    Raises:
        FileNotFoundError: Nếu chưa có mô hình.
    """
    if not DECISION_TREE_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Chưa có mô hình Decision Tree. Hãy huấn luyện trước."
        )
    if not RANDOM_FOREST_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Chưa có mô hình Random Forest. Hãy huấn luyện trước."
        )

    dt_model = joblib.load(DECISION_TREE_MODEL_FILE)
    rf_model = joblib.load(RANDOM_FOREST_MODEL_FILE)

    le = None
    if LABEL_ENCODER_FILE.exists():
        le = joblib.load(LABEL_ENCODER_FILE)

    return dt_model, rf_model, le


def predict_one(
    features: dict,
    score_averages: Optional[list[float]] = None,
    comment_statuses: Optional[list[str]] = None,
) -> dict:
    """
    Dự đoán kết quả học tập cho một học sinh.

    Args:
        features: Dictionary chứa các feature values.
        score_averages: Danh sách ĐTB các môn (cho Rule Engine).
        comment_statuses: Danh sách trạng thái nhận xét (cho Rule Engine).

    Returns:
        Dictionary chứa:
        - rule_engine_result: Kết quả Rule Engine
        - dt_prediction: Dự đoán Decision Tree
        - rf_prediction: Dự đoán Random Forest
        - final_prediction: Kết quả cuối cùng
        - confidence: Độ tin cậy
        - rule_reason: Lý do Rule Engine
    """
    dt_model, rf_model, le = load_trained_models()

    # Chuẩn bị input cho ML
    feature_values = [features.get(col, 0) for col in FEATURE_COLUMNS]
    X_input = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

    # ML predictions
    dt_pred_encoded = dt_model.predict(X_input)[0]
    rf_pred_encoded = rf_model.predict(X_input)[0]

    # Decode predictions
    if le is not None:
        dt_prediction = le.inverse_transform([dt_pred_encoded])[0]
        rf_prediction = le.inverse_transform([rf_pred_encoded])[0]
    else:
        dt_prediction = CLASS_LABELS[dt_pred_encoded] if isinstance(dt_pred_encoded, (int, np.integer)) else str(dt_pred_encoded)
        rf_prediction = CLASS_LABELS[rf_pred_encoded] if isinstance(rf_pred_encoded, (int, np.integer)) else str(rf_pred_encoded)

    # Confidence (probability from Random Forest)
    rf_proba = rf_model.predict_proba(X_input)[0]
    confidence = round(float(max(rf_proba)) * 100, 1)

    # Probability cho từng class
    if le is not None:
        class_proba = {le.inverse_transform([i])[0]: round(float(p) * 100, 1) for i, p in enumerate(rf_proba)}
    else:
        class_proba = {CLASS_LABELS[i]: round(float(p) * 100, 1) for i, p in enumerate(rf_proba)}

    # Rule Engine result
    rule_result = None
    rule_reason = "Không đủ dữ liệu để chạy Rule Engine."
    if score_averages is not None and comment_statuses is not None:
        rule_result = classify_learning_result(score_averages, comment_statuses)
        rule_reason = get_classification_reason(score_averages, comment_statuses, rule_result)

    # Final prediction: Hybrid decision (ML + Safeguard BGDDT)
    # Lớp bảo vệ kiến trúc (Architectural Safeguard): Áp dụng các luật loại trừ bắt buộc của Bộ GD&ĐT
    comment_fail = features.get("comment_not_pass_count", 0)
    score_lt_3_5 = features.get("count_score_lt_3_5", 0)
    attendance = features.get("attendance_rate", 100)

    if comment_fail >= 2 or score_lt_3_5 > 0 or attendance < 75:
        final_prediction = "Chưa đạt"
    else:
        final_prediction = rf_prediction

    return {
        "rule_engine_result": rule_result,
        "dt_prediction": dt_prediction,
        "rf_prediction": rf_prediction,
        "final_prediction": final_prediction,
        "confidence": confidence,
        "class_probabilities": class_proba,
        "rule_reason": rule_reason,
        "features": features,
    }


def predict_batch(
    df: pd.DataFrame,
    score_averages_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Dự đoán hàng loạt cho DataFrame.

    Args:
        df: DataFrame chứa features.

    Returns:
        DataFrame gốc với các cột dự đoán bổ sung.
    """
    dt_model, rf_model, le = load_trained_models()

    # Validate
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()

    # Predictions
    dt_preds_encoded = dt_model.predict(X)
    rf_preds_encoded = rf_model.predict(X)

    # Decode
    if le is not None:
        dt_preds = le.inverse_transform(dt_preds_encoded)
        rf_preds = le.inverse_transform(rf_preds_encoded)
    else:
        dt_preds = [CLASS_LABELS[p] if isinstance(p, (int, np.integer)) else str(p) for p in dt_preds_encoded]
        rf_preds = [CLASS_LABELS[p] if isinstance(p, (int, np.integer)) else str(p) for p in rf_preds_encoded]

    # Confidence
    rf_proba = rf_model.predict_proba(X)
    confidences = [round(float(max(proba)) * 100, 1) for proba in rf_proba]

    # Build result
    result_df = df.copy()
    result_df["dt_prediction"] = dt_preds
    result_df["rf_prediction"] = rf_preds
    
    # Final prediction: Hybrid decision (ML + Safeguard BGDDT)
    final_preds = []
    for idx, row in result_df.iterrows():
        comment_fail = row.get("comment_not_pass_count", 0)
        score_lt_3_5 = row.get("count_score_lt_3_5", 0)
        attendance = row.get("attendance_rate", 100)
        if comment_fail >= 2 or score_lt_3_5 > 0 or attendance < 75:
            final_preds.append("Chưa đạt")
        else:
            final_preds.append(row["rf_prediction"])
            
    result_df["final_prediction"] = final_preds
    result_df["confidence"] = confidences

    # Tích hợp Early Warning System phân tích chỉ số rủi ro
    from src.early_warning import analyze_batch_ews
    result_df = analyze_batch_ews(result_df)

    return result_df

