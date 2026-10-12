# src/generation/generator.py

import ollama

MODEL_NAME = "llama3:8b"

def generate_rag_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Nhận vào câu hỏi và danh sách các chunk tài liệu đã truy xuất,
    sau đó gọi Ollama để sinh câu trả lời bám sát ngữ cảnh kèm trích dẫn nguồn.
    """
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        doc_id = chunk.get('document_id', 'unknown_doc')
        content = chunk.get('content', '')
        context_str += f"[Citation ID: {doc_id} - Chunk {i+1}]\n{content}\n\n"

    system_prompt = (
        "Bạn là một trợ lý AI doanh nghiệp thông minh. "
        "Hãy trả lời câu hỏi dựa CHÍNH XÁC VÀO các tài liệu tham khảo được cung cấp dưới đây. "
        "Tuyệt đối không tự suy diễn hoặc bịa đặt thông tin ngoài tài liệu. "
        "Nếu tài liệu không đủ thông tin để trả lời, hãy trả lời chính xác câu: "
        "'Hiện tại hệ thống không đủ thông tin để trả lời câu hỏi này.' "
        "Mỗi thông tin quan trọng phải đính kèm mã trích dẫn nguồn (ví dụ: [document_id])."
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
            options={'temperature': 0.1}
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"Lỗi khi gọi Ollama API sinh câu trả lời: {str(e)}"