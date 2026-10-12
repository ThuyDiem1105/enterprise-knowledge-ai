"""
main.py
Script điều phối toàn bộ pipeline tự động: 
1. Ingestion (src/ingestion/pipeline.py)
2. Chunking (src/chunking/chunker.py)
3. Embedding & Vector Store (src/embeddings/embedder.py)
Chỉ cần chạy file này để thực thi toàn bộ quy trình chuẩn bị dữ liệu.
"""

import os
import json
from pathlib import Path

# Import các module có sẵn trong cấu trúc src/ của project
from src.ingestion.pipeline import IngestionPipeline
from src.chunking.chunker import RecursiveTextChunker
from src.embeddings.embedder import TextEmbedder

def main():
    print("==================================================")
    print(" BẮT ĐẦU CHẠY TOÀN BỘ QUY TRÌNH PIPELINE TỰ ĐỘNG")
    print("==================================================")
    
    # 1. Xác định đường dẫn tuyệt đối theo thư mục gốc
    BASE_DIR = Path(__file__).resolve().parent
    target_file_path = BASE_DIR / "data" / "raw" / "Luat_An_Toan_Thong_Tin_Mang.pdf"
    
    # Fallback kiểm tra nếu file tên là download.pdf
    if not target_file_path.exists():
        target_file_path = BASE_DIR / "data" / "raw" / "download.pdf"
        
    if not target_file_path.exists():
        print(f"[Lỗi nghiêm trọng] Không tìm thấy file PDF tài liệu trong thư mục 'data/raw/'.")
        print("Vui lòng đặt file PDF vào thư mục data/raw/ trước khi chạy.")
        return

    print(f"\n[1/3] Đang xử lý Ingestion tài liệu: {target_file_path.name}")
    ingestion_pipeline = IngestionPipeline(processed_data_dir=str(BASE_DIR / "data" / "processed"))
    
    ingestion_result = ingestion_pipeline.process_document(
        file_path=str(target_file_path), 
        access_scope="internal"
    )
    
    if not ingestion_result:
        print("[Lỗi] Quá trình Ingestion thất bại.")
        return
        
    doc_record = ingestion_result["document_record"]
    extracted_text = ingestion_result["extracted_text"]
    print(f" -> Thành công! Document ID: {doc_record['document_id']}")
    print(f" -> Tổng số ký tự trích xuất: {len(extracted_text)} ký tự")

    print(f"\n[2/3] Đang thực hiện Chunking văn bản...")
    chunker = RecursiveTextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split_text(
        text=extracted_text, 
        document_id=doc_record["document_id"], 
        base_metadata=doc_record["metadata"]
    )
    print(f" -> Thành công! Đã cắt tài liệu thành {len(chunks)} chunks.")

    print(f"\n[3/3] Đang chạy Embedding & Lưu trữ Vector Store...")
    # Khởi tạo embedder từ module src/embeddings/embedder.py có sẵn
    embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Sinh vector cho toàn bộ chunks
    embedded_chunks = embedder.encode_chunks(chunks)
    
    # Lưu kết quả vector vào thư mục dữ liệu vector_store
    vector_dir = BASE_DIR / "data" / "vector_store"
    vector_dir.mkdir(parents=True, exist_ok=True)
    vector_file = vector_dir / "embedded_chunks.json"
    
    with open(vector_file, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
        
    print(f" -> Thành công! Đã lưu vector index vào: {vector_file}")
    print("\n==================================================")
    print(" HOÀN TẤT TOÀN BỘ QUY TRÌNH CHUẨN BỊ DỮ LIỆU THÀNH CÔNG!")
    print("==================================================")

if __name__ == "__main__":
    main()