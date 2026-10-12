"""
src/parsing/base_parser.py
Định nghĩa Abstract Base Class cho các module parser tài liệu (PDF, DOCX, HTML, Markdown).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDocumentParser(ABC):
    
    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Phương thức bắt buộc các parser con phải hiện thực hóa.
        Đầu vào: Đường dẫn tuyệt đối hoặc tương đối tới file thô.
        Đầu ra: Dictionary chứa:
            - text: Nội dung toàn văn bản trích xuất
            - metadata: Thông tin chi tiết (số trang, tác giả, tiêu đề, cấu trúc)
            - success: Trạng thái thành công hay thất bại (True/False)
            - error: Thông báo lỗi nếu có
        """
        pass