"""
src/embeddings/embedder.py
Module chuyển đổi văn bản (chunks) thành các vector biểu diễn sử dụng Sentence Transformers.
"""

from typing import List, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class TextEmbedder:
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Khởi tạo mô hình embedding. 
        Mặc định sử dụng 'all-MiniLM-L6-v2' (nhanh, nhẹ, hiệu quả cao cho tiếng Việt và tiếng Anh).
        """
        self.model_name = model_name
        if SentenceTransformer is not None:
            print(f"Đang tải mô hình embedding: {model_name}...")
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
            print("[Cảnh báo] Thư viện sentence_transformers chưa được cài đặt.")

    def encode_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Nhận vào danh sách các chunk văn bản, tính toán vector embedding và gắn trực tiếp vào mỗi chunk.
        """
        if self.model is None:
            raise RuntimeError("Mô hình Embedding chưa được khởi tạo do thiếu thư viện.")
            
        if not chunks:
            return []

        # Trích xuất danh sách nội dung văn bản từ các chunk
        texts = [chunk["content"] for chunk in chunks]
        
        # Thực hiện batch embedding
        print(f"Đang tính toán vector embedding cho {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Gắn vector kết quả vào từng chunk dictionary
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            # Chuyển đổi numpy array thành list để dễ dàng lưu trữ vào JSON hoặc Vector DB
            chunk_copy["embedding"] = embeddings[i].tolist()
            embedded_chunks.append(chunk_copy)
            
        print("[Thành công] Đã hoàn tất việc sinh embedding cho các chunks.")
        return embedded_chunks


if __name__ == "__main__":
    # Đoạn code kiểm thử chạy độc lập module embedder (cần cài đặt sentence-transformers)
    print("Module TextEmbedder đã sẵn sàng.")