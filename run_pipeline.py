"""
run_pipeline.py
Script thực thi kết nối từ Ingestion tài liệu thực tế (PDF) đến Chunker (Đã tối ưu đường dẫn an toàn).
"""

import os
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline
from src.chunking.chunker import RecursiveTextChunker

def main():
    # Tự động xác định thư mục gốc của project dựa vào vị trí file run_pipeline.py hiện tại
    BASE_DIR = Path(__file__).resolve().parent
    
    # Đường dẫn tuyệt đối tới file PDF trong thư mục data/raw/
    target_file_path = BASE_DIR / "data" / "raw" / "download.pdf"
    
    # Kiểm tra nếu chưa có file mẫu thì báo hướng dẫn
    if not target_file_path.exists():
        print(f"[Thông báo] Chưa tìm thấy file tại '{target_file_path}'.")
        print("Vui lòng kiểm tra lại tên file hoặc thư mục data/raw/.")
        return

    print(f"--- ĐÃ TÌM THẤY FILE: {target_file_path.name} ---")
    print("--- BƯỚC 1: ĐANG CHẠY INGESTION PIPELINE ---")
    
    ingestion_pipeline = IngestionPipeline(processed_data_dir=str(BASE_DIR / "data" / "processed"))
    
    # Xử lý tài liệu (trích xuất text, tính checksum SHA-256, sinh document_id thật)
    ingestion_result = ingestion_pipeline.process_document(
        file_path=str(target_file_path), 
        access_scope="internal"
    )
    
    if not ingestion_result:
        print("[Lỗi] Quá trình Ingestion thất bại.")
        return
        
    doc_record = ingestion_result["document_record"]
    extracted_text = ingestion_result["extracted_text"]
    
    print(f"\nĐã xử lý thành công tài liệu:")
    print(f" - Document ID thật: {doc_record['document_id']}")
    print(f" - Tên file: {doc_record['filename']}")
    print(f" - Tổng số ký tự trích xuất: {len(extracted_text)} ký tự")

    print("\n--- BƯỚC 2: ĐANG CHẠY CHUNKING PIPELINE ---")
    # Khởi tạo Chunker với kích thước chunk chuẩn
    chunker = RecursiveTextChunker(chunk_size=500, chunk_overlap=50)
    
    # Thực hiện chia đoạn văn bản thực tế từ tài liệu
    chunks = chunker.split_text(
        text=extracted_text, 
        document_id=doc_record["document_id"], 
        base_metadata=doc_record["metadata"]
    )
    
    print(f"\n[Hoàn tất] Tài liệu Luật thực tế đã được cắt thành {len(chunks)} chunks chuẩn bị cho bước Embedding!")
    
    # In thử thông tin chunk đầu tiên để kiểm tra
    if chunks:
        print("\nVí dụ Chunk đầu tiên:")
        print(f"ID: {chunks[0]['chunk_id']}")
        print(f"Metadata trang: {chunks[0]['metadata'].get('total_pages', 'N/A')}")
        print(f"Nội dung: {chunks[0]['content'][:200]}...")

if __name__ == "__main__":
    main()