"""
explanations.py — Giải thích kết quả dự đoán bằng tiếng Việt.

Module này cung cấp:
- Giải thích Rule Engine
- Giải thích ML predictions
- Tóm tắt Decision Tree
- Giải thích Feature Importance
"""

from typing import Optional
import pandas as pd
import numpy as np

from src.config import FEATURE_COLUMNS


# Tên tiếng Việt cho từng feature
FEATURE_NAMES_VI = {
    "avg_score": "Điểm trung bình",
    "min_score": "Điểm thấp nhất",
    "max_score": "Điểm cao nhất",
    "std_score": "Độ lệch chuẩn điểm",
    "count_score_ge_8": "Số môn ĐTB ≥ 8.0",
    "count_score_ge_6_5": "Số môn ĐTB ≥ 6.5",
    "count_score_ge_5": "Số môn ĐTB ≥ 5.0",
    "count_score_lt_3_5": "Số môn ĐTB < 3.5",
    "comment_not_pass_count": "Số môn nhận xét Chưa đạt",
    "attendance_rate": "Tỷ lệ chuyên cần (%)",
    "assignment_completion_rate": "Tỷ lệ hoàn thành bài tập (%)",
    "participation_score": "Điểm tham gia",
    "behavior_score": "Điểm hành vi",
    "teacher_evaluation_score": "Điểm đánh giá giáo viên",
    "progress_delta": "Mức tiến bộ",
}


def generate_ml_explanation(
    features: dict,
    prediction: str,
    model_name: str = "Random Forest",
) -> str:
    """
    Sinh giải thích cho dự đoán ML dựa trên feature values.

    Args:
        features: Dictionary feature values của học sinh.
        prediction: Nhãn dự đoán.
        model_name: Tên mô hình.

    Returns:
        Chuỗi giải thích tiếng Việt.
    """
    explanations = []
    explanations.append(f"**Mô hình {model_name} dự đoán: {prediction}**\n")
    explanations.append("Phân tích các yếu tố chính:\n")

    avg_score = features.get("avg_score", 0)
    min_score = features.get("min_score", 0)
    attendance = features.get("attendance_rate", 0)
    comment_fail = features.get("comment_not_pass_count", 0)
    count_ge_8 = features.get("count_score_ge_8", 0)
    count_lt_3_5 = features.get("count_score_lt_3_5", 0)
    behavior = features.get("behavior_score", 0)
    participation = features.get("participation_score", 0)

    # Phân tích điểm số
    if avg_score >= 8.0:
        explanations.append(f"✅ Điểm trung bình cao: **{avg_score}**")
    elif avg_score >= 6.5:
        explanations.append(f"📊 Điểm trung bình khá: **{avg_score}**")
    elif avg_score >= 5.0:
        explanations.append(f"⚠️ Điểm trung bình trung bình: **{avg_score}**")
    else:
        explanations.append(f"❌ Điểm trung bình thấp: **{avg_score}**")

    if min_score < 3.5:
        explanations.append(f"❌ Có môn điểm rất thấp: **{min_score}**")
    elif min_score < 5.0:
        explanations.append(f"⚠️ Điểm thấp nhất: **{min_score}**")

    if count_ge_8 >= 6:
        explanations.append(f"✅ Có **{count_ge_8}** môn đạt từ 8.0 trở lên")
    elif count_ge_8 >= 3:
        explanations.append(f"📊 Có **{count_ge_8}** môn đạt từ 8.0")

    if count_lt_3_5 > 0:
        explanations.append(f"❌ Có **{count_lt_3_5}** môn dưới 3.5")

    # Phân tích nhận xét
    if comment_fail > 0:
        explanations.append(f"⚠️ Có **{comment_fail}** môn nhận xét Chưa đạt")
    else:
        explanations.append("✅ Tất cả môn nhận xét đều Đạt")

    # Phân tích chuyên cần
    if attendance >= 90:
        explanations.append(f"✅ Chuyên cần tốt: **{attendance}%**")
    elif attendance >= 75:
        explanations.append(f"📊 Chuyên cần khá: **{attendance}%**")
    elif attendance >= 60:
        explanations.append(f"⚠️ Chuyên cần cần cải thiện: **{attendance}%**")
    else:
        explanations.append(f"❌ Chuyên cần thấp: **{attendance}%**")

    # Phân tích hành vi
    if behavior >= 7.5:
        explanations.append(f"✅ Hành vi tốt: **{behavior}/10**")
    elif behavior < 5.0:
        explanations.append(f"⚠️ Hành vi cần cải thiện: **{behavior}/10**")

    return "\n".join(explanations)


def explain_feature_importance(
    importances: list[float],
    feature_names: list[str],
    top_n: int = 5,
) -> str:
    """
    Giải thích feature importance bằng tiếng Việt.

    Args:
        importances: Danh sách importance values.
        feature_names: Tên features.
        top_n: Số features quan trọng nhất cần giải thích.

    Returns:
        Chuỗi giải thích.
    """
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    lines = ["**Các yếu tố quan trọng nhất ảnh hưởng đến phân loại:**\n"]

    for i, (name, imp) in enumerate(pairs[:top_n], 1):
        vi_name = FEATURE_NAMES_VI.get(name, name)
        lines.append(f"{i}. **{vi_name}** — mức ảnh hưởng: {imp:.1%}")

    return "\n".join(lines)


def explain_decision_tree_summary(dt_model) -> str:
    """
    Tóm tắt Decision Tree bằng tiếng Việt.

    Args:
        dt_model: Mô hình Decision Tree đã train.

    Returns:
        Chuỗi tóm tắt.
    """
    tree = dt_model.tree_
    lines = [
        "**Tóm tắt Decision Tree:**\n",
        f"- Độ sâu tối đa: **{tree.max_depth}**",
        f"- Số nút (nodes): **{tree.node_count}**",
        f"- Số lá (leaves): **{tree.n_leaves}**",
        "",
        "Decision Tree hoạt động bằng cách chia dữ liệu thành các nhóm nhỏ hơn",
        "dựa trên ngưỡng của từng feature. Mỗi nút là một câu hỏi dạng",
        '"Feature X <= ngưỡng?" và mỗi lá là một nhãn phân loại.',
    ]
    return "\n".join(lines)


def explain_rule_engine_logic() -> str:
    """Trả về giải thích logic Rule Engine dạng markdown."""
    return """
### 📐 Logic Rule Engine theo Bộ GD&ĐT

Rule Engine **không phải Machine Learning**. Đây là bộ luật cố định dùng để phân loại học sinh.

#### Công thức tính điểm

**DTB_mhk** (Điểm trung bình môn học kỳ):
```
DTB_mhk = (ΣĐiểm_TX + 2 × Điểm_GK + 3 × Điểm_CK) / (Số_điểm_TX + 5)
```

**DTB_mcn** (Điểm trung bình môn cả năm):
```
DTB_mcn = (DTB_HK1 + 2 × DTB_HK2) / 3
```

#### Luật phân loại

| Mức | Điều kiện |
|---|---|
| 🟢 **Tốt** | Tất cả nhận xét Đạt + Tất cả ĐTB ≥ 6.5 + ≥ 6 môn ĐTB ≥ 8.0 |
| 🔵 **Khá** | Tất cả nhận xét Đạt + Tất cả ĐTB ≥ 5.0 + ≥ 6 môn ĐTB ≥ 6.5 |
| 🟡 **Đạt** | ≤ 1 nhận xét Chưa đạt + ≥ 6 môn ĐTB ≥ 5.0 + Không có ĐTB < 3.5 |
| 🔴 **Chưa đạt** | Các trường hợp còn lại |

#### Vai trò trong hệ thống

Rule Engine vừa là **logic phân loại chính thức**, vừa là **baseline** để tạo nhãn `learning_result_label` cho supervised learning. Machine Learning học từ nhãn này.
"""
