"""
src/parsing/pdf_parser.py
Parser chuyên dụng để bóc tách nội dung và metadata từ tài liệu PDF bằng PyMuPDF.
"""

import pymupdf as fitz
from typing import Dict, Any
from src.parsing.base_parser import BaseDocumentParser


class PDFParser(BaseDocumentParser):
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Thực hiện đọc file PDF, trích xuất text từng trang và metadata gốc của tài liệu.
        """
        try:
            doc = fitz.open(file_path)
            full_text_list = []
            pages_metadata = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                full_text_list.append(text)
                
                # Lưu thông tin chi tiết từng trang phục vụ việc trích xuất nguồn sau này
                pages_metadata.append({
                    "page_number": page_num + 1,
                    "char_count": len(text)
                })
                
            metadata = {
                "total_pages": len(doc),
                "pages": pages_metadata,
                "format": "pdf",
                "pdf_metadata": doc.metadata
            }
            
            return {
                "text": "\n".join(full_text_list),
                "metadata": metadata,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "text": "",
                "metadata": {},
                "success": False,
                "error": str(e)
            }