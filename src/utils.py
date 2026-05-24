"""
utils.py — Các hàm tiện ích dùng chung trong dự án.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any


def safe_round(value: float, decimals: int = 1) -> float:
    """Làm tròn an toàn, xử lý NaN."""
    if pd.isna(value):
        return 0.0
    return round(float(value), decimals)


def parse_regular_scores(scores_str: str) -> list[float]:
    """
    Parse chuỗi điểm thường xuyên thành list float.
    
    Ví dụ: "7.5;8.0;8.5" → [7.5, 8.0, 8.5]

    Args:
        scores_str: Chuỗi điểm phân tách bằng dấu chấm phẩy.

    Returns:
        Danh sách điểm float.
    """
    if pd.isna(scores_str) or str(scores_str).strip() == "":
        return []
    
    parts = str(scores_str).split(";")
    result = []
    for part in parts:
        part = part.strip()
        if part:
            try:
                result.append(float(part))
            except ValueError:
                continue
    return result


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """Giới hạn giá trị trong khoảng [min_val, max_val]."""
    return max(min_val, min(max_val, float(value)))


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format số thành chuỗi phần trăm."""
    return f"{round(value, decimals)}%"


def generate_vietnamese_name(index: int) -> str:
    """
    Sinh tên tiếng Việt giả lập cho học sinh.
    
    Args:
        index: Số thứ tự học sinh.
    
    Returns:
        Tên học sinh dạng "Học sinh XXX".
    """
    return f"Học sinh {index:03d}"


def get_label_color(label: str) -> str:
    """Trả về mã màu CSS cho nhãn phân loại."""
    colors = {
        "Tốt": "#10B981",
        "Khá": "#3B82F6",
        "Đạt": "#F59E0B",
        "Chưa đạt": "#EF4444",
    }
    return colors.get(label, "#6B7280")


def get_label_emoji(label: str) -> str:
    """Trả về emoji cho nhãn phân loại."""
    emojis = {
        "Tốt": "🟢",
        "Khá": "🔵",
        "Đạt": "🟡",
        "Chưa đạt": "🔴",
    }
    return emojis.get(label, "⚪")
