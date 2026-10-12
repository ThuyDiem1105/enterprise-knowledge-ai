"""
src/chunking/chunker.py
Module chia nhỏ văn bản tối ưu, an toàn tuyệt đối chống vòng lặp vô tận.
"""

from typing import List, Dict, Any


class RecursiveTextChunker:
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, document_id: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text:
            return []

        chunks = []
        text_length = len(text)
        start_idx = 0
        chunk_index = 0

        # Đảm bảo vòng lặp luôn tiến về phía trước, tránh vòng lặp vô tận
        while start_idx < text_length:
            end_idx = min(start_idx + self.chunk_size, text_length)
            
            chunk_content = text[start_idx:end_idx].strip()
            
            if chunk_content:
                chunk_id = f"{document_id}_chunk_{chunk_index:03d}"
                
                chunk_meta = base_metadata.copy() if base_metadata else {}
                chunk_meta["chunk_id"] = chunk_id
                chunk_meta["document_id"] = document_id
                chunk_meta["chunk_index"] = chunk_index
                chunk_meta["char_length"] = len(chunk_content)

                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "content": chunk_content,
                    "metadata": chunk_meta
                })
                
                chunk_index += 1

            # Điểm mấu chốt chống lỗi lặp vô tận: 
            # Nếu end_idx đã đến cuối văn bản thì dừng, ngược lại bắt buộc phải dịch chuyển tiến lên
            if end_idx >= text_length:
                break
                
            next_start = end_idx - self.chunk_overlap
            # Nếu overlap đẩy start_idx lùi về phía sau hoặc đứng yên, ép buộc tiến lên để tránh kẹt
            if next_start <= start_idx:
                start_idx = end_idx
            else:
                start_idx = next_start

        print(f"[Thành công] Đã chia tài liệu '{document_id}' thành {len(chunks)} chunks một cách an toàn.")
        return chunks