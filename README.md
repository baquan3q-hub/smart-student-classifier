# 🎓 Smart Student Classification System

## Hệ thống phân loại kết quả học tập học sinh thông minh

### 📋 Giới thiệu

Đây là hệ thống demo học thuật phục vụ báo cáo cuối kỳ. Hệ thống phân loại kết quả học tập học sinh theo 4 mức **Tốt / Khá / Đạt / Chưa đạt** dựa trên:

1. **Rule Engine** — theo quy định Bộ GD&ĐT Việt Nam
2. **Machine Learning** — Decision Tree và Random Forest (học máy có giám sát)

### 🚀 Cài đặt và chạy

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Chạy ứng dụng
streamlit run app.py
```

### 📁 Cấu trúc dự án

```
smart-student-classifier/
├── app.py                  # Ứng dụng Streamlit chính
├── requirements.txt        # Dependencies
├── data/                   # Dữ liệu
│   ├── raw/                # Dữ liệu thô
│   ├── processed/          # Dữ liệu đã xử lý
│   └── subject_config.json # Cấu hình môn học
├── models/                 # Mô hình đã huấn luyện
├── reports/                # Báo cáo kết quả
├── src/                    # Source code
│   ├── config.py           # Cấu hình
│   ├── data_generator.py   # Tạo dữ liệu mô phỏng
│   ├── rule_engine.py      # Rule Engine BGDDT
│   ├── feature_engineering.py # Kỹ thuật đặc trưng
│   ├── preprocessing.py    # Tiền xử lý dữ liệu
│   ├── train_models.py     # Huấn luyện mô hình
│   ├── evaluation.py       # Đánh giá mô hình
│   ├── predict.py          # Dự đoán
│   ├── explanations.py     # Giải thích kết quả
│   ├── recommendations.py  # Khuyến nghị
│   ├── visualization.py    # Biểu đồ
│   └── utils.py            # Tiện ích
└── assets/                 # Tài nguyên
```

### 🎯 Chức năng chính

| Chức năng | Mô tả |
|---|---|
| Tạo dataset | Tạo dữ liệu mô phỏng 300+ học sinh |
| Rule Engine | Tính DTB_mhk, DTB_mcn, phân loại theo Bộ GD&ĐT |
| Huấn luyện ML | Decision Tree + Random Forest |
| Dự đoán | Dự đoán 1 học sinh hoặc hàng loạt (upload CSV) |
| Giải thích | Lý do phân loại + khuyến nghị hỗ trợ |

### ⚠️ Lưu ý quan trọng

- **Dữ liệu hoàn toàn mô phỏng**, không sử dụng thông tin học sinh thật.
- Hệ thống chỉ mang tính **demo học thuật**, không thay thế đánh giá của giáo viên.
- Không sử dụng Deep Learning hay Neural Network.

### 📊 Nhãn phân loại

| Nhãn | Ý nghĩa |
|---|---|
| **Tốt** | Kết quả học tập xuất sắc |
| **Khá** | Kết quả học tập tốt |
| **Đạt** | Kết quả học tập đạt yêu cầu |
| **Chưa đạt** | Cần hỗ trợ thêm |

### 🔧 Công nghệ sử dụng

- Python 3.9+
- Streamlit
- scikit-learn
- pandas, numpy
- matplotlib, seaborn, plotly

### 👤 Tác giả

Dự án báo cáo cuối kỳ — Môn Học máy
