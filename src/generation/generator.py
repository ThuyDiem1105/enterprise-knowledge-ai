# ==========================================
# FILE: src/generation/generator.py
# Chức năng: Xử lý phần sinh câu trả lời kèm bắt buộc trích dẫn (Giai đoạn 5)
# ==========================================

import ollama

MODEL_NAME = "llama3:8b"

def generate_rag_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Nhận vào câu hỏi và danh sách các chunk tài liệu đã truy xuất,
    bắt buộc mô hình phải chèn mã document_id vào câu trả lời.
    """
    # Xây dựng context rõ ràng, nhấn mạnh document_id cho mô hình thấy
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        doc_id = chunk.get('document_id', 'unknown_doc')
        content = chunk.get('content', '')
        context_str += f"--- NGUỒN TÀI LIỆU [ID: {doc_id}] ---\n{content}\n\n"

    system_prompt = (
        "Bạn là một trợ lý AI doanh nghiệp thông minh. "
        "BẮT BUỘC phải trả lời bằng TIẾNG VIỆT mạch lạc, súc tích. "
        "QUY TẮC CỐT LÕI (ƯU TIÊN SỐ 1): "
        "1. Hãy kiểm tra kỹ xem các tài liệu tham khảo có chứa thông tin thực sự liên quan để trả lời câu hỏi hay không. "
        "2. Nếu tài liệu KHÔNG ĐỦ THÔNG TIN hoặc KHÔNG CÓ CÂU TRẢ LỜI cho câu hỏi, BẮT BUỘC phải trả lời nguyên văn câu sau và không nói gì thêm: "
        "'Hiện tại hệ thống không đủ thông tin để trả lời câu hỏi này.' "
        "3. Nếu có thông tin, hãy trả lời dựa trên tài liệu và PHẢI đính kèm mã ID của nguồn đó ngay sau câu theo định dạng [ID: document_id] (Ví dụ: [ID: doc_79a87e2c7e17]). "
        "Tuyệt đối không tự bịa đặt thông tin ngoài tài liệu."
    )

    user_prompt = f"""
    Tài liệu tham khảo:
    {context_str}

    Câu hỏi của người dùng: {query}
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': 0.1,
                'repeat_penalty': 1.2,
                'num_predict': 350
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"Lỗi khi gọi Ollama API sinh câu trả lời: {str(e)}"