"""
src/ingestion/pipeline.py
Pipeline xử lý và chuẩn hóa tài liệu đầu vào: chống trùng lặp qua checksum, 
quản lý trạng thái xử lý và kiểm định schema bằng Pydantic.
"""

import hashlib
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from src.parsing.pdf_parser import PDFParser


class DocumentSchema(BaseModel):
    document_id: str = Field(..., description="Mã định danh duy nhất của tài liệu dựa trên hash")
    filename: str = Field(..., description="Tên file gốc của tài liệu")
    content_hash: str = Field(..., description="Mã SHA-256 checksum đảm bảo không trùng lặp nội dung")
    source_uri: str = Field(..., description="Đường dẫn nguồn lưu trữ tài liệu")
    created_at: str = Field(..., description="Thời gian tạo bản ghi (ISO format)")
    updated_at: str = Field(..., description="Thời gian cập nhật bản ghi gần nhất")
    parser_version: str = Field(default="1.0.0", description="Phiên bản của parser đã sử dụng")
    processing_status: str = Field(default="completed", description="Trạng thái: pending, completed, failed")
    access_scope: str = Field(default="internal", description="Phạm vi phân quyền: public, internal, confidential")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata chi tiết từ parser (số trang, cấu trúc)")


class IngestionPipeline:
    
    def __init__(self, processed_data_dir: str = "./data/processed"):
        self.processed_data_dir = processed_data_dir
        os.makedirs(processed_data_dir, exist_ok=True)
        # Khởi tạo các parser hỗ trợ
        self.pdf_parser = PDFParser()

    def calculate_file_hash(self, file_path: str) -> str:
        """
        Tính toán mã SHA-256 checksum của file để nhận diện file trùng lặp tuyệt đối.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def process_document(self, file_path: str, access_scope: str = "internal") -> Optional[Dict[str, Any]]:
        """
        Thực hiện toàn bộ quy trình ingestion cho một file tài liệu thô.
        """
        if not os.path.exists(file_path):
            print(f"[Lỗi] Không tìm thấy file tại đường dẫn: {file_path}")
            return None
            
        filename = os.path.basename(file_path)
        content_hash = self.calculate_file_hash(file_path)
        
        # Tạo document_id ổn định dựa trên 12 ký tự đầu của chuỗi hash nội dung
        document_id = f"doc_{content_hash[:12]}"
        
        # Nhận diện định dạng file qua phần mở rộng
        file_extension = filename.split(".")[-1].lower()
        
        parse_result = {}
        if file_extension == "pdf":
            parse_result = self.pdf_parser.parse(file_path)
        else:
            print(f"[Cảnh báo] Định dạng file .{file_extension} chưa được hỗ trợ.")
            return None
            
        if not parse_result.get("success", False):
            print(f"[Lỗi] Trích xuất file {filename} thất bại: {parse_result.get('error')}")
            return None
            
        current_time = datetime.utcnow().isoformat()
        
        # Đóng gói thông tin tài liệu theo chuẩn Pydantic Schema
        doc_record = DocumentSchema(
            document_id=document_id,
            filename=filename,
            content_hash=content_hash,
            source_uri=file_path,
            created_at=current_time,
            updated_at=current_time,
            parser_version="1.0.0",
            processing_status="completed",
            access_scope=access_scope,
            metadata=parse_result.get("metadata", {})
        )
        
        print(f"[Thành công] Đã ingest tài liệu: {filename} (ID: {document_id})")
        
        return {
            "document_record": doc_record.model_dump(),
            "extracted_text": parse_result.get("text")
        }


if __name__ == "__main__":
    # Đoạn code kiểm thử ví dụ chạy độc lập pipeline
    pipeline = IngestionPipeline()
    print("Module Ingestion Pipeline đã sẵn sàng vận hành.")