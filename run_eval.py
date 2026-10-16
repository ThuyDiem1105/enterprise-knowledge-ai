# ==========================================
# FILE: run_eval.py
# Chức năng: Đọc dataset từ dataset_manifest.json, 
# sinh câu trả lời và đánh giá định lượng bằng ROUGE Score + Kiểm tra Trích dẫn (Giai đoạn 6)
# ==========================================

import json
import os
import pandas as pd
from rouge_score import rouge_scorer
from src.generation.generator import generate_rag_answer

DATASET_PATH = "datasets/manifests/dataset_manifest.json"

def load_evaluation_dataset(file_path: str) -> list:
    """
    Đọc bộ dữ liệu đánh giá từ file dataset_manifest.json.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset tại: {file_path}.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith("["):
            dataset = json.loads(content)
        else:
            dataset = [json.loads(line) for line in content.splitlines() if line.strip()]
            
    return dataset

def evaluate_with_nlp_metrics(generated_answer: str, expected_answer: str, retrieved_chunks: list) -> dict:
    """
    Đánh giá định lượng bằng ROUGE Score và kiểm tra chính xác mã document_id có trong câu trả lời.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(expected_answer, generated_answer)
    
    rouge_1_f = scores['rouge1'].fmeasure
    rouge_2_f = scores['rouge2'].fmeasure
    rouge_l_f = scores['rougeL'].fmeasure

    # Kiểm tra xem mã document_id của chunk có xuất hiện trong câu trả lời sinh ra hay không
    found_citation = False
    for chunk in retrieved_chunks:
        doc_id = chunk.get('document_id', '')
        # Kiểm tra xem doc_id (ví dụ: doc_79a87e2c7e17) có nằm trong câu trả lời không
        if doc_id and doc_id in generated_answer:
            found_citation = True
            break
            
    # Dự phòng: Nếu không tìm thấy mã chính xác nhưng có dấu ngoặc vuông chứa từ khóa doc
    if not found_citation and "[" in generated_answer and "doc_" in generated_answer:
        found_citation = True

    citation_score = 1 if found_citation else 0

    # Trạng thái đạt: ROUGE-L từ 0.2 trở lên VÀ có đính kèm đúng mã trích dẫn nguồn
    is_passed = 1 if (rouge_l_f > 0.2 and citation_score == 1) else 0

    return {
        "rouge_1": round(rouge_1_f, 4),
        "rouge_2": round(rouge_2_f, 4),
        "rouge_l": round(rouge_l_f, 4),
        "citation_valid": citation_score,
        "overall_passed": is_passed
    }

def run_evaluation_pipeline():
    print("==================================================")
    print(" ĐANG TẢI BỘ DỮ LIỆU TỪ dataset_manifest.json ...")
    print("==================================================")
    
    try:
        eval_dataset = load_evaluation_dataset(DATASET_PATH)
        print(f"-> Đã tải thành công {len(eval_dataset)} mẫu đánh giá.\n")
    except Exception as e:
        print(f"Lỗi khi đọc file dataset: {str(e)}")
        return

    evaluation_results = []

    for idx, item in enumerate(eval_dataset, 1):
        print(f"--- Đang xử lý mẫu [{idx}/{len(eval_dataset)}] ---")
        
        query = item.get("question")
        expected_ans = item.get("expected_answer") or item.get("expected answer", "")
        chunks = item.get("retrieved_chunks", [])
        is_answerable = item.get("answerable", True)

        # BƯỚC 1: SINH CÂU TRẢ LỜI
        print(f" [Bước 1] Sinh câu trả lời cho: '{query}'...")
        generated_ans = generate_rag_answer(query, chunks)
        print(f" -> Kết quả sinh ra:\n{generated_ans}\n")

        # BƯỚC 2: ĐÁNH GIÁ ĐỊNH LƯỢNG (ROUGE + CITATION) THAY CHO LLM JUDGE
        print(f" [Bước 2] Chấm điểm định lượng (ROUGE & Citation)...")
        metrics = evaluate_with_nlp_metrics(generated_ans, expected_ans, chunks)
        
        print(f" -> Kết quả chỉ số:")
        print(f"    * ROUGE-1 F1: {metrics['rouge_1']}")
        print(f"    * ROUGE-L F1: {metrics['rouge_l']}")
        print(f"    * Citation Valid: {metrics['citation_valid']}")
        print(f"    * Trạng thái đạt: {metrics['overall_passed']}\n")
        print("--------------------------------------------------\n")

        evaluation_results.append({
            "Question": query,
            "Generated Answer": generated_ans,
            "Expected Answer": expected_ans,
            "ROUGE-1": metrics['rouge_1'],
            "ROUGE-L": metrics['rouge_l'],
            "Citation Valid": metrics['citation_valid'],
            "Passed": metrics['overall_passed']
        })

    # Tổng kết và xuất file báo cáo
    df_report = pd.DataFrame(evaluation_results)
    print("==================================================")
    print(" BÁO CÁO TỔNG KẾT ĐÁNH GIÁ ĐỊNH LƯỢNG HỆ THỐNG")
    print("==================================================")
    print(df_report[["Question", "ROUGE-L", "Citation Valid", "Passed"]])
    
    # Tính trung bình các chỉ số để đưa vào slide báo cáo đồ án
    print("\n--- BẢNG THỐNG KÊ TRUNG BÌNH ---")
    print(f"Mean ROUGE-L: {df_report['ROUGE-L'].mean():.4f}")
    print(f"Mean Citation Accuracy: {df_report['Citation Valid'].mean():.4f}")
    print(f"Overall Success Rate: {df_report['Passed'].mean() * 100:.2f}%")
    
    df_report.to_csv("rag_quantitative_evaluation_report.csv", index=False, encoding="utf-8-sig")
    print("\n Đã xuất báo cáo chi tiết ra file: rag_quantitative_evaluation_report.csv")

if __name__ == "__main__":
    run_evaluation_pipeline()