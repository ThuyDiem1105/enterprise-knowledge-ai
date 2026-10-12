"""
apps/api/main.py
Xây dựng RESTful API cho Enterprise Knowledge AI Assistant sử dụng FastAPI,
hỗ trợ quản lý tài liệu, phân quyền truy cập và endpoint hỏi đáp có trích dẫn.
"""
import sys
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path để Python nhận diện được package apps/src
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="Enterprise Knowledge AI Assistant API",
    version="1.0.0",
    description="API hệ thống AI tìm kiếm, truy xuất và trả lời câu hỏi dựa trên tài liệu nội bộ doanh nghiệp."
)


# --- Định nghĩa Schema dữ liệu cho API request/response ---
class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi từ người dùng dưới ngôn ngữ tự nhiên")
    user_role: str = Field(default="internal", description="Vai trò người dùng để kiểm tra phân quyền: public, internal, confidential")


class CitationDetail(BaseModel):
    chunk_id: str
    document_id: str
    content_snippet: str


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời từ AI kèm theo mã trích dẫn")
    citations: List[str] = Field(..., description="Danh sách mã ID tài liệu bằng chứng")
    status: str = Field(..., description="Trạng thái xử lý: success, abstain, error")


# --- Các API Endpoints ---
@app.get("/health", summary="Kiểm tra trạng thái hệ thống")
def health_check():
    return {"status": "healthy", "service": "Enterprise Knowledge AI API"}


@app.post("/v1/chat", response_model=ChatResponse, summary="Hỏi đáp có trích dẫn nguồn")
def chat_endpoint(request: ChatRequest):
    """
    Endpoint tiếp nhận câu hỏi của người dùng, thực hiện truy xuất tài liệu theo phân quyền 
    và sinh câu trả lời Grounded RAG.
    """
    user_role = request.user_role
    query = request.query
    
    # Kiểm tra phân quyền giả lập (trong thực tế sẽ lọc vector store theo access_scope)
    if not query.strip():
        raise HTTPException(status_code=400, description="Câu hỏi không được để trống.")
        
    # Mô phỏng kết quả trả về từ RAG Pipeline hoàn chỉnh
    simulated_answer = (
        f"Dựa trên tài liệu nội bộ, chính sách liên quan đến yêu cầu của bạn được quy định rõ ràng. "
        f"[Source ID: policy-document-01]"
    )
    
    return ChatResponse(
        answer=simulated_answer,
        citations=["policy-document-01"],
        status="success"
    )


@app.get("/v1/documents", summary="Liệt kê danh sách tài liệu được phép truy cập")
def list_documents(user_role: str = Header(default="internal", description="Vai trò phân quyền của người dùng")):
    """
    Trả về danh sách các tài liệu mà user_role có quyền đọc.
    """
    mock_documents = [
        {"document_id": "doc_a1b2c3", "filename": "Chính_sách_lưu_trữ_dữ_liệu.pdf", "access_scope": "internal"},
        {"document_id": "doc_d4e5f6", "filename": "Quy_chế_làm_việc_từ_xa.pdf", "access_scope": "internal"},
        {"document_id": "doc_7890ab", "filename": "Báo_cáo_tài_chính_mật.pdf", "access_scope": "confidential"}
    ]
    
    # Lọc tài liệu theo phân quyền
    if user_role == "public":
        filtered_docs = [d for d in mock_documents if d["access_scope"] == "public"]
    elif user_role == "internal":
        filtered_docs = [d for d in mock_documents if d["access_scope"] in ["public", "internal"]]
    else: # confidential / admin
        filtered_docs = mock_documents
        
    return {"user_role": user_role, "documents": filtered_docs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)