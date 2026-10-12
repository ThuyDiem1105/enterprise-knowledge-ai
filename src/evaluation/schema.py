"""
src/evaluation/schema.py
Định nghĩa Pydantic schema để kiểm tra và chuẩn hóa cấu trúc dữ liệu đánh giá RAG (tiếng Việt).
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class EvaluationRecord(BaseModel):
    question_id: str = Field(..., description="Mã định danh duy nhất cho câu hỏi (ví dụ: Q001)")
    question: str = Field(..., description="Câu hỏi của người dùng bằng tiếng Việt")
    expected_answer: str = Field(..., description="Đáp án mẫu chuẩn được trích xuất từ tài liệu nội bộ")
    expected_source_ids: List[str] = Field(
        ..., description="Danh sách ID các tài liệu chứa đoạn văn bản làm bằng chứng"
    )
    answerable: bool = Field(
        ..., description="True nếu tài liệu có chứa câu trả lời, False nếu ngoài phạm vi tài liệu"
    )
    category: str = Field(
        ..., description="Phân loại câu hỏi: direct (trực tiếp), multi-hop (tổng hợp), unanswerable (ngoài phạm vi)"
    )


class EvaluationDatasetSchema(BaseModel):
    records: List[EvaluationRecord]


if __name__ == "__main__":
    import json
    
    # Script kiểm tra đọc file manifest tiếng Việt bằng Pydantic schema
    manifest_path = "datasets/manifests/dataset_manifest.json"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Validate toàn bộ danh sách qua Pydantic
        validated_dataset = [EvaluationRecord(**item) for item in data]
        print(f"Thành công! Đã load và validate hợp lệ {len(validated_dataset)} câu hỏi tiếng Việt từ {manifest_path}.")
    except Exception as e:
        print(f"Lỗi validate dữ liệu benchmark: {e}")