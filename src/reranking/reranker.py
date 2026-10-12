"""
src/reranking/reranker.py
Module Reranker sử dụng Cross-Encoder để chấm điểm và sắp xếp lại 
các kết quả từ bộ Hybrid Search, đảm bảo độ chính xác cao nhất trước khi đưa vào LLM.
"""
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để Python nhận diện được package src
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class DocumentReranker:
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Khởi tạo mô hình Cross-Encoder Reranker.
        """
        self.model_name = model_name
        self.encoder = CrossEncoder(model_name)
        print(f"[Khởi tạo] Đã nạp thành công mô hình Cross-Encoder: {model_name}")

    def rerank(self, query: str, retrieved_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Chấm điểm lại danh sách chunks dựa trên độ tương quan trực tiếp giữa câu hỏi (query) và nội dung chunk.
        """
        if not retrieved_chunks:
            return []

        # Tạo các cặp [query, chunk_content] để Cross-Encoder đánh giá đồng thời
        pairs = [[query, chunk["content"]] for chunk in retrieved_chunks]
        
        # Dự đoán điểm số liên quan
        scores = self.encoder.predict(pairs)
        
        # Gắn điểm số rerank vào từng chunk
        for i, chunk in enumerate(retrieved_chunks):
            chunk["rerank_score"] = float(scores[i])
            
        # Sắp xếp lại danh sách theo điểm rerank giảm dần
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # Trả về top_k kết quả tốt nhất cuối cùng
        return sorted_chunks[:top_k]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Tự động thêm thư mục gốc vào đường dẫn hệ thống để chạy trực tiếp không bị lỗi import
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
        
    from src.retrieval.hybrid_search import HybridRetriever
    
    vector_path = ROOT_DIR / "data" / "vector_store" / "embedded_chunks.json"
    if vector_path.exists():
        retriever = HybridRetriever(vector_store_path=str(vector_path))
        reranker = DocumentReranker()
        
        sample_query = "Hành vi nào bị nghiêm cấm về an toàn thông tin mạng"
        
        # Lấy top 6 từ Hybrid Search
        hybrid_results = retriever.retrieve(sample_query, top_k=6)
        
        # Rerank lại lấy top 3 chính xác nhất
        final_results = reranker.rerank(sample_query, hybrid_results, top_k=3)
        
        print(f"\n--- KẾT QUẢ SAU KHI RERANK CHO CÂU HỎI: '{sample_query}' ---")
        for i, res in enumerate(final_results, 1):
            print(f"{i}. [Chunk ID: {res['chunk_id']}] (Rerank Score: {res.get('rerank_score', 0):.4f})")
            print(f"   Nội dung: {res['content'][:150]}...\n")
    else:
        print(f"Chưa tìm thấy file vector tại {vector_path}. Vui lòng chạy main.py trước.")