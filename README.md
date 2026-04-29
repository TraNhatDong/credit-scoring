# 🏦 Enterprise Credit Scoring System

![Microservices](https://img.shields.io/badge/Architecture-Microservices-blue)
![Java](https://img.shields.io/badge/Java-25-orange)
![Spring Boot](https://img.shields.io/badge/Spring_Boot-3-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Ready-blue)

Hệ thống Chấm điểm Tín dụng (Credit Scoring) tự động đánh giá rủi ro hồ sơ vay vốn của khách hàng dựa trên dữ liệu lịch sử và các chỉ số tài chính. Hệ thống được thiết kế theo tiêu chuẩn phần mềm doanh nghiệp, tích hợp Trí tuệ nhân tạo có thể giải thích (Explainable AI - XAI) để đảm bảo tính minh bạch trong các quyết định tài chính.

## 🌟 Tính năng nổi bật
* **Đánh giá rủi ro thời gian thực:** Dự đoán tỷ lệ nợ xấu bằng các mô hình học máy mạnh mẽ (XGBoost/Random Forest).
* **AI Giải thích được (XAI):** Tích hợp thư viện SHAP để phân tích và xuất báo cáo lý do cấu thành điểm tín dụng (yếu tố nào cộng điểm, yếu tố nào trừ điểm).
* **Kiến trúc Microservices:** Tách biệt hoàn toàn luồng xử lý AI (Python) và luồng nghiệp vụ lõi (Java).
* **Giao diện hiện đại:** Giao diện quản trị áp dụng phong cách Glassmorphism chuyên nghiệp, trực quan hóa dữ liệu rủi ro.

## 🛠 Tech Stack

### AI Service (Phân tích dữ liệu & Mô hình)
* **Framework:** FastAPI (Python 3.10+)
* **Machine Learning:** Scikit-learn, XGBoost
* **Explainable AI:** SHAP
* **Data Processing:** Pandas, SMOTE (Xử lý mất cân bằng dữ liệu)

### Core Backend (Nghiệp vụ lõi)
* **Ngôn ngữ & Framework:** Java 25, Spring Boot 3
* **Database:** PostgreSQL (Sử dụng chuẩn 3NF và cột `JSONB` để lưu Audit Log, SHAP response)
* **Tích hợp:** RESTful API, Global Exception Handling

### Deployment & DevOps
* Container hóa bằng Docker & Docker Compose.

## 📂 Cấu trúc thư mục (Folder Structure)

```text
enterprise-credit-scoring/
├── ai-service/                # Python FastAPI Microservice
│   ├── app/                   # API logic & Pydantic models
│   ├── models/                # File .pkl/.onnx chứa mô hình đã train
│   ├── notebooks/             # Jupyter notebooks (EDA, Training)
│   ├── requirements.txt       
│   └── Dockerfile
├── core-backend/              # Java Spring Boot Microservice
│   ├── src/main/java/         # Controllers, Services, Repositories
│   ├── src/main/resources/    # application.yml
│   ├── pom.xml
│   └── Dockerfile
├── frontend/                  # Giao diện người dùng
│   ├── src/                   # React/Vue components (Glassmorphism UI)
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml         # File khởi chạy toàn bộ hệ thống
└── README.md