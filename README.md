# 🏢 Enterprise Knowledge AI Assistant

Hệ thống Trợ lý AI doanh nghiệp thông minh dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)**, tích hợp Local LLM (qua Ollama), cơ chế phân quyền truy cập dữ liệu đa tầng, kiểm chứng trích dẫn nguồn chuẩn xác và bộ công cụ đánh giá định lượng tự động.

## 📌 Tổng quan Dự án
Trong môi trường doanh nghiệp hiện đại, việc tra cứu khối lượng lớn tài liệu nội bộ (chính sách, quy chế nhân sự, báo cáo kỹ thuật) thường tốn nhiều thời gian và dễ xảy ra tình trạng thông tin không đồng nhất. Dự án **Enterprise Knowledge AI Assistant** được xây dựng nhằm giải quyết bài toán này với các mục tiêu cốt lõi:
* **Độ chính xác cao & Chống bịa đặt (Anti-Hallucination):** Mô hình bắt buộc phải trả lời dựa trên ngữ cảnh tài liệu được truy xuất và có cơ chế từ chối thông minh (`Abstention`) khi dữ liệu không đủ.
* **Trích dẫn nguồn minh bạch:** Mọi câu trả lời đều được đính kèm mã định danh nguồn tài liệu (`document_id`) giúp người dùng dễ dàng đối chiếu.
* **Bảo mật & Phân quyền:** Hỗ trợ kiểm soát quyền truy cập tài liệu theo vai trò người dùng (`public`, `internal`, `confidential`).
* **Hiệu suất tối ưu & Chi phí thấp:** Chạy hoàn toàn cục bộ (Local LLM) qua Ollama, bảo mật tuyệt đối dữ liệu nội bộ doanh nghiệp.

## 🛠️ Tech Stack (Công nghệ sử dụng)
* **Ngôn ngữ lập trình:** Python 3.10+
* **LLM Engine:** Ollama (Local execution với *Llama 3:8b* hoặc *Llama 3.2:3b*)
* **Backend API:** FastAPI, Pydantic, Uvicorn
* **Frontend UI:** Streamlit (Hỗ trợ chat tương tác và Streaming response)
* **Data Processing & Evaluation:** Pandas, PyPDF, Rouge-Score (Đánh giá định lượng ROUGE-1, ROUGE-L)

## 📂 Cấu trúc Thư mục Dự án
```text
D:\enterprise-knowledge-ai
|
├── apps/
│   ├── api/main.py              # FastAPI Backend (Quản lý API, phân quyền, chat endpoint)
│   └── web/app.py               # Streamlit Frontend (Giao diện người dùng trực quan)
├── src/
│   └── generation/generator.py  # Xử lý sinh câu trả lời RAG, trích dẫn & chống lặp từ
├── dataset_manifest.json        # Bộ dữ liệu chuẩn đánh giá (Evaluation Dataset)
├── run_eval.py                  # Script chạy đánh giá định lượng (ROUGE & Citation Valid)
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Tài liệu hướng dẫn dự án
```
## 🚀 Hướng dẫn Cài đặt và Triển khai
1. Cài đặt Python 3.10+ và Ollama (`ollama pull llama3:8b` hoặc `ollama pull llama3.2:3b`).
2. Tạo môi trường ảo và cài đặt thư viện:
   python -m venv venv
   venv\Scripts\activate
   pip install fastapi uvicorn streamlit ollama pandas rouge-score requests

## 🖥️ Hướng dẫn Vận hành Hệ thống
- **Bước 1 (Backend):** uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
- **Bước 2 (Frontend):** streamlit run apps/web/app.py
- **Bước 3 (Evaluation):** python run_eval.py
README.md - Raw Source Code
