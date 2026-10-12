"""
src/retrieval/hybrid_search.py
Module tìm kiếm hỗn hợp (Hybrid Search) kết hợp Dense Vector Search và Keyword Search (BM25),
được tổng hợp bằng thuật toán Reciprocal Rank Fusion (RRF).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class HybridRetriever:
    
    def __init__(self, vector_store_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Khởi tạo Hybrid Retriever từ file Vector Store đã lưu ở bước trước.
        """
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        
        # Tải danh sách embedded chunks từ file JSON
        if not Path(vector_store_path).exists():
            raise FileNotFoundError(f"Không tìm thấy file vector store tại: {vector_store_path}")
            
        with open(vector_store_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        # Chuẩn bị dữ liệu cho BM25
        self.corpus = [chunk["content"] for chunk in self.chunks]
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Chuẩn bị ma trận embedding cho Dense Search
        self.embeddings = np.array([chunk["embedding"] for chunk in self.chunks])
        
        print(f"[Khởi tạo] Đã nạp thành công {len(self.chunks)} chunks vào Hybrid Search Engine.")

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm ngữ nghĩa dựa trên vector embedding (Cosine Similarity)."""
        query_vector = self.encoder.encode(query, convert_to_numpy=True)
        
        # Tính độ tương đồng Cosine
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1e-10
        scores = np.dot(self.embeddings, query_vector) / norms
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)
            
        return results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm từ khóa truyền thống bằng thuật toán BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Chỉ lấy các kết quả có khớp từ khóa
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                results.append(chunk)
                
        return results[:top_k]

    def rrf(self, dense_results: List[Dict], bm25_results: List[Dict], k: int = 60, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Thuật toán Reciprocal Rank Fusion (RRF) để kết hợp hạng từ Dense Search và BM25.
        Công thức: RRF_score = sum(1 / (k + rank))
        """
        rrf_scores = {}
        chunk_map = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        
        # Tính điểm từ Dense Search
        for rank, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            
        # Tính điểm từ BM25 Search
        for rank, chunk in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            
        # Sắp xếp lại theo điểm RRF giảm dần
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for cid, score in sorted_chunks[:top_k]:
            chunk = chunk_map[cid].copy()
            chunk["rrf_score"] = score
            final_results.append(chunk)
            
        return final_results

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Thực hiện tìm kiếm kết hợp Hybrid Search hoàn chỉnh."""
        # Lấy gấp đôi ứng viên từ mỗi nhánh để thuật toán RRF tổng hợp chính xác hơn
        dense_res = self.dense_search(query, top_k=top_k * 2)
        bm25_res = self.keyword_search(query, top_k=top_k * 2)
        
        # Tổng hợp bằng RRF
        return self.rrf(dense_res, bm25_res, top_k=top_k)


if __name__ == "__main__":
    # Test thử nghiệm độc lập module Hybrid Search
    vector_path = "data/vector_store/embedded_chunks.json"
    if Path(vector_path).exists():
        retriever = HybridRetriever(vector_store_path=vector_path)
        sample_query = "An toàn thông tin mạng là gì"
        results = retriever.retrieve(sample_query, top_k=3)
        
        print(f"\n--- KẾT QUẢ TEST HYBRID SEARCH CHO CÂU HỎI: '{sample_query}' ---")
        for i, res in enumerate(results, 1):
            print(f"{i}. [Chunk ID: {res['chunk_id']}] (RRF Score: {res.get('rrf_score', 0):.4f})")
            print(f"   Nội dung: {res['content'][:150]}...\n")
    else:
        print(f"Chưa tìm thấy file vector tại {vector_path}. Vui lòng chạy main.py trước.")