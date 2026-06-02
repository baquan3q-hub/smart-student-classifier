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

    # Tính độ tin cậy đồng thuận kết hợp (Consensus Confidence Score)
    consensus_conf = confidence
    consensus_status = "Đồng thuận trung bình"
    
    if rule_result is not None:
        if rule_result == dt_prediction == rf_prediction == final_prediction:
            consensus_conf = min(100.0, confidence + 10.0)
            consensus_status = "Đồng thuận hoàn toàn"
        else:
            status_parts = []
            if dt_prediction != rf_prediction:
                consensus_conf -= 10.0
                status_parts.append("ML lệch nhau (-10%)")
            priority = {"Chưa đạt": 0, "Đạt": 1, "Khá": 2, "Tốt": 3}
            rule_pri = priority.get(rule_result, 0)
            final_pri = priority.get(final_prediction, 0)
            if rule_pri < final_pri:
                consensus_conf -= 15.0
                status_parts.append("Luật Bộ GD&ĐT thấp hơn ML (-15%)")
            elif rule_result != final_prediction:
                consensus_conf -= 10.0
                status_parts.append("Luật Bộ GD&ĐT lệch với ML (-10%)")
            
            consensus_conf = max(0.0, consensus_conf)
            consensus_status = f"Mâu thuẫn: {', '.join(status_parts)}" if status_parts else "Đồng thuận cục bộ"
    else:
        if dt_prediction == rf_prediction == final_prediction:
            consensus_conf = min(100.0, confidence + 5.0)
            consensus_status = "Đồng thuận ML"
        else:
            consensus_conf = max(0.0, confidence - 10.0)
            consensus_status = "ML lệch nhau (-10%)"

    consensus_confidence = round(consensus_conf, 1)

    return {
        "rule_engine_result": rule_result,
        "dt_prediction": dt_prediction,
        "rf_prediction": rf_prediction,
        "final_prediction": final_prediction,
        "confidence": confidence,
        "consensus_confidence": consensus_confidence,
        "consensus_status": consensus_status,
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
    
    # Final prediction & Consensus Confidence calculation
    final_preds = []
    consensus_confidences = []
    consensus_statuses = []
    rule_engine_results = []

    for idx, row in result_df.iterrows():
        comment_fail = int(row.get("comment_not_pass_count", 0))
        score_lt_3_5 = int(row.get("count_score_lt_3_5", 0))
        attendance = row.get("attendance_rate", 100)

        if comment_fail >= 2 or score_lt_3_5 > 0 or attendance < 75:
            final_pred = "Chưa đạt"
        else:
            final_pred = row["rf_prediction"]
        final_preds.append(final_pred)

        # Ước lượng kết quả Rule Engine
        avg_score = row.get("avg_score", 0.0)
        min_score = row.get("min_score", 0.0)
        score_avgs = [avg_score, min_score] + [avg_score] * 6
        comment_stats = ["Đạt"] * max(0, 3 - comment_fail) + ["Chưa đạt"] * comment_fail
        
        rule_res = classify_learning_result(score_avgs, comment_stats)
        rule_engine_results.append(rule_res)

        dt_pred = row["dt_prediction"]
        rf_pred = row["rf_prediction"]
        conf = confidences[idx]

        # Tính toán độ tin cậy đồng thuận kết hợp
        consensus_conf = conf
        consensus_status = "Đồng thuận trung bình"

        if rule_res == dt_pred == rf_pred == final_pred:
            consensus_conf = min(100.0, conf + 10.0)
            consensus_status = "Đồng thuận hoàn toàn"
        else:
            status_parts = []
            if dt_pred != rf_pred:
                consensus_conf -= 10.0
                status_parts.append("ML lệch nhau (-10%)")
            priority = {"Chưa đạt": 0, "Đạt": 1, "Khá": 2, "Tốt": 3}
            rule_pri = priority.get(rule_res, 0)
            final_pri = priority.get(final_pred, 0)
            if rule_pri < final_pri:
                consensus_conf -= 15.0
                status_parts.append("Luật Bộ GD&ĐT thấp hơn ML (-15%)")
            elif rule_res != final_pred:
                consensus_conf -= 10.0
                status_parts.append("Luật Bộ GD&ĐT lệch với ML (-10%)")

            consensus_conf = max(0.0, consensus_conf)
            consensus_status = f"Mâu thuẫn: {', '.join(status_parts)}" if status_parts else "Đồng thuận cục bộ"

        consensus_confidences.append(round(consensus_conf, 1))
        consensus_statuses.append(consensus_status)

    result_df["rule_engine_result"] = rule_engine_results
    result_df["final_prediction"] = final_preds
    result_df["confidence"] = confidences
    result_df["consensus_confidence"] = consensus_confidences
    result_df["consensus_status"] = consensus_statuses

    # Tích hợp Early Warning System phân tích chỉ số rủi ro
    from src.early_warning import analyze_batch_ews
    result_df = analyze_batch_ews(result_df)

    return result_df

