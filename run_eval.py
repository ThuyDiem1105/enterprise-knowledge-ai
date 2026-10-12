# ==========================================
# FILE: run_eval.py
# Chức năng: Chạy pipeline đánh giá, LLM-as-a-Judge và Error Analysis (Giai đoạn 6)
# ==========================================

import json
import ollama
import pandas as pd
from src.generation.generator import generate_rag_answer  # Import module sinh câu trả lời ở File 1

MODEL_NAME = "llama3:8b"

# Bộ dữ liệu đánh giá giả lập (Chuẩn bị theo Giai đoạn 1 của project)
EVAL_DATASET = [
    {
        "question": "Chính sách lưu trữ dữ liệu hệ thống của công ty là gì?",
        "expected_answer": "Chính sách yêu cầu giữ log hệ thống trong vòng 90 ngày kể từ ngày phát sinh.",
        "retrieved_chunks": [
            {
                "document_id": "policy-document-01",
                "content": "Chính sách lưu trữ dữ liệu của công ty yêu cầu giữ log hệ thống trong vòng 90 ngày kể từ ngày phát sinh trên hệ thống lưu trữ chính."
            }
        ],
        "answerable": True
    },
    {
        "question": "Quy trình xin nghỉ phép năm diễn ra như thế nào?",
        "expected_answer": "Nhân viên phải nộp đơn trước 3 ngày làm việc qua cổng thông tin nội bộ.",
        "retrieved_chunks": [
            {
                "document_id": "hr-manual-05",
                "content": "Quy định chung về nhân sự: Mọi nhân viên được hưởng 12 ngày phép năm theo luật lao động."
            }
        ],
        "answerable": False  # Chunk không chứa thông tin về quy trình xin nghỉ phép
    }
]

def evaluate_answer_with_judge(query: str, retrieved_context: str, generated_answer: str, expected_answer: str, answerable: bool) -> dict:
    """
    Đóng vai trò LLM-as-a-Judge để chấm điểm tính trung thực (Faithfulness) 
    và phát hiện lỗi cụ thể của câu trả lời vừa sinh ra.
    """
    judge_prompt = f"""
    Bạn là một chuyên gia đánh giá hệ thống RAG doanh nghiệp khó tính.
    Hãy phân tích và đánh giá câu trả lời của hệ thống dựa trên các tiêu chí sau:

    1. [Faithfulness / Tính trung thực]: Câu trả lời có hoàn toàn dựa trên ngữ cảnh được cung cấp hay không? Có thông tin nào bịa đặt không? (Điểm: 1 nếu chuẩn xác, 0 nếu bịa đặt/sai lệch).
    2. [Abstention Check / Khả năng từ chối]: Nếu câu hỏi không thể trả lời bằng ngữ cảnh (answerable = {answerable}), hệ thống có từ chối đúng cách không? (Điểm: 1 nếu xử lý đúng, 0 nếu sai).
    3. [Error Identification / Phát hiện lỗi]: Chỉ ra cụ thể mô hình sai ở đâu (ví dụ: bịa thông tin ngoài ngữ cảnh, không chịu trích dẫn nguồn, từ chối nhầm, hoặc suy diễn sai).

    [Dữ liệu đầu vào]:
    - Câu hỏi: {query}
    - Ngữ cảnh được truy xuất: {retrieved_context}
    - Đáp án tham chiếu: {expected_answer}
    - Câu trả lời của hệ thống: {generated_answer}

    Hãy trả về kết quả DUY NHẤT dưới dạng chuỗi JSON thuần túy theo cấu trúc sau (không kèm markdown format ngoài):
    {{
      "faithfulness_score": 0 hoặc 1,
      "abstention_score": 0 hoặc 1,
      "error_type": "Không có lỗi" hoặc mô tả ngắn gọn lỗi gặp phải,
      "detailed_analysis": "Phân tích chi tiết bằng tiếng Việt về nguyên nhân tại sao đúng hoặc sai."
    }}
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': judge_prompt}],
            options={'temperature': 0.0} # Giám khảo cần khách quan tuyệt đối
        )
        
        raw_content = response['message']['content'].strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
            
        result = json.loads(raw_content.strip())
        return result
    except Exception as e:
        return {
            "faithfulness_score": 0,
            "abstention_score": 0,
            "error_type": "JSON Parse Error",
            "detailed_analysis": f"Không thể parse JSON từ Judge. Lỗi: {str(e)}. Raw output: {response['message']['content']}"
        }

def run_evaluation_pipeline():
    print("==================================================")
    print(" BẮT ĐẦU CHẠY PIPELINE: SINH TEXT -> ĐÁNH GIÁ (LOCAL)")
    print("==================================================\n")

    evaluation_results = []

    for idx, item in enumerate(EVAL_DATASET, 1):
        print(f"--- Đang xử lý câu hỏi [{idx}/{len(EVAL_DATASET)}] ---")
        query = item["question"]
        expected_ans = item["expected_answer"]
        chunks = item["retrieved_chunks"]
        is_answerable = item["answerable"]

        context_text = "\n".join([c["content"] for c in chunks])

        # BƯỚC 1: GỌI MODULE GENERATOR ĐỂ SINH CÂU TRẢ LỜI TRƯỚC
        print(f" [Bước 1] Sinh câu trả lời cho: '{query}'...")
        generated_ans = generate_rag_answer(query, chunks)
        print(f" -> Kết quả sinh ra:\n{generated_ans}\n")

        # BƯỚC 2: GỌI LLM JUDGE ĐỂ ĐÁNH GIÁ SAU ĐÓ VÀ TÌM LỖI
        print(f" [Bước 2] Đánh giá bằng LLM-as-a-Judge và phân tích lỗi...")
        evaluation = evaluate_answer_with_judge(
            query=query,
            retrieved_context=context_text,
            generated_answer=generated_ans,
            expected_answer=expected_ans,
            answerable=is_answerable
        )
        
        print(f" -> Kết quả chấm điểm:")
        print(f"    * Faithfulness Score: {evaluation.get('faithfulness_score')}")
        print(f"    * Abstention Score: {evaluation.get('abstention_score')}")
        print(f"    * Loại lỗi phát hiện: {evaluation.get('error_type')}")
        print(f"    * Phân tích chi tiết: {evaluation.get('detailed_analysis')}\n")
        print("--------------------------------------------------\n")

        evaluation_results.append({
            "Question": query,
            "Generated Answer": generated_ans,
            "Expected Answer": expected_ans,
            "Faithfulness": evaluation.get("faithfulness_score"),
            "Abstention Correct": evaluation.get("abstention_score"),
            "Error Type": evaluation.get("error_type"),
            "Analysis": evaluation.get("detailed_analysis")
        })

    # Tổng kết và xuất file báo cáo
    df_report = pd.DataFrame(evaluation_results)
    print("==================================================")
    print(" BÁO CÁO TỔNG KẾT ĐÁNH GIÁ HỆ THỐNG")
    print("==================================================")
    print(df_report[["Question", "Faithfulness", "Error Type"]])
    
    df_report.to_csv("rag_evaluation_report.csv", index=False, encoding="utf-8-sig")
    print("\n Đã xuất báo cáo chi tiết ra file: rag_evaluation_report.csv")

if __name__ == "__main__":
    run_evaluation_pipeline()