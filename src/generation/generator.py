"""
src/generation/generator.py
Module chịu trách nhiệm sinh câu trả lời Grounded RAG sử dụng Ollama (Local LLM),
hỗ trợ trích dẫn nguồn chuẩn xác, chống bịa đặt và tối ưu hóa độ trễ (latency).
"""

import ollama
from typing import List, Dict, Generator, Union

# Cấu hình mô hình: Bạn có thể đổi sang 'llama3.2:3b' nếu muốn tốc độ nhanh hơn nữa
MODEL_NAME = "llama3:8b"

def generate_rag_answer(query: str, retrieved_chunks: List[Dict], stream: bool = False) -> Union[str, Generator[str, None, None]]:
    """
    Nhận vào câu hỏi và danh sách các chunk tài liệu đã được truy xuất,
    sau đó gọi Ollama để sinh câu trả lời bám sát ngữ cảnh kèm trích dẫn nguồn.
    
    Args:
        query (str): Câu hỏi của người dùng.
        retrieved_chunks (List[Dict]): Danh sách các chunk (chứa 'document_id' và 'content').
        stream (bool): Nếu True, trả về generator để hiển thị chữ dạng streaming.
        
    Returns:
        str hoặc Generator: Câu trả lời hoàn chỉnh hoặc luồng stream các token.
    """
    # 1. Xây dựng context rõ ràng, làm nổi bật document_id cho mô hình
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        doc_id = chunk.get('document_id', f'unknown_doc_{i}')
        content = chunk.get('content', '').strip()
        context_str += f"--- NGUỒN TÀI LIỆU [ID: {doc_id}] ---\n{content}\n\n"

    # 2. Thiết kế System Prompt chuẩn RAG doanh nghiệp
    system_prompt = (
        "Bạn là một trợ lý AI doanh nghiệp thông minh, chuyên gia trả lời dựa trên tài liệu nội bộ. "
        "BẮT BUỘC phải trả lời bằng TIẾNG VIỆT mạch lạc, súc tích. "
        "\n\nQUY TẮC CỐT LÕI (ƯU TIÊN SỐ 1):"
        "\n1. Hãy kiểm tra kỹ xem các tài liệu tham khảo có chứa thông tin thực sự liên quan để trả lời câu hỏi hay không."
        "\n2. Nếu tài liệu KHÔNG ĐỦ THÔNG TIN hoặc KHÔNG CÓ CÂU TRẢ LỜI cho câu hỏi, BẮT BUỘC phải trả lời nguyên văn câu sau và không nói gì thêm: "
        "'Hiện tại hệ thống không đủ thông tin để trả lời câu hỏi này.'"
        "\n3. Nếu có thông tin, hãy trả lời dựa trên tài liệu và PHẢI đính kèm mã ID của nguồn đó ngay sau ý được sử dụng theo định dạng [ID: document_id] (Ví dụ: [ID: doc_79a87e2c7e17])."
        "\n4. Tuyệt đối không tự bịa đặt thông tin ngoài tài liệu. Tuyệt đối không lặp lại các ý hoặc gạch đầu dòng giống hệt nhau nhiều lần."
    )

    user_prompt = f"""
    Tài liệu tham khảo:
    {context_str}

    Câu hỏi của người dùng: {query}
    """

    # Cấu hình tối ưu độ trễ và chống lặp từ
    generation_options = {
        'temperature': 0.1,        # Độ sáng tạo thấp để ép mô hình bám sát tài liệu
        'repeat_penalty': 1.2,     # Phạt nặng các từ/câu bị lặp lại (chống lỗi lặp dòng)
        'num_predict': 350         # Giới hạn số lượng token tối đa để tăng tốc độ phản hồi
    }

    try:
        if stream:
            # Trả về luồng streaming để hiển thị mượt mà trên giao diện
            response_stream = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options=generation_options,
                stream=True
            )
            def token_generator():
                for chunk in response_stream:
                    content_chunk = chunk.get('message', {}).get('content', '')
                    if content_chunk:
                        yield content_chunk
            return token_generator()
        else:
            # Trả về kết quả một lần hoàn chỉnh
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options=generation_options
            )
            return response.get('message', {}).get('content', '').strip()

    except Exception as e:
        error_msg = f"Lỗi khi kết nối hoặc gọi Ollama API: {str(e)}"
        if stream:
            def error_gen():
                yield error_msg
            return error_gen()
        return error_msg