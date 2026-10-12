"""
src/evaluation/evaluator.py
Khung đánh giá tự động (Evaluation Framework) cho hệ thống RAG doanh nghiệp,
giúp đo lường các chỉ số Recall@k, độ chính xác tìm kiếm và theo dõi thực nghiệm.
"""

import json
from typing import List, Dict, Any


class RAGEvaluator:
    
    def __init__(self, manifest_path: str = "datasets/manifests/dataset_manifest.json"):
        """
        Khởi tạo Evaluator bằng cách nạp bộ dữ liệu benchmark chuẩn (manifest).
        """
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.evaluation_dataset = json.load(f)
            print(f"[Thành công] Đã nạp {len(self.evaluation_dataset)} câu hỏi benchmark từ {manifest_path}")
        except Exception as e:
            self.evaluation_dataset = []
            print(f"[Lỗi] Không thể nạp file dataset manifest: {e}")

    def evaluate_retrieval_recall(self, system_retrieval_results: Dict[str, List[str]], k: int = 3) -> float:
        """
        Tính toán chỉ số Recall@k cho phần Retrieval.
        - system_retrieval_results: Dictionary dạng {question_id: [danh_sách_source_id_được_trích_xuất]}
        - k: Giới hạn số lượng top chunk lấy ra để kiểm tra
        """
        if not self.evaluation_dataset:
            return 0.0

        hits = 0
        total_answerable = 0

        for item in self.evaluation_dataset:
            # Chỉ đánh giá trên các câu hỏi có đáp án trong tài liệu (answerable = True)
            if not item.get("answerable", True):
                continue
                
            total_answerable += 1
            question_id = item["question_id"]
            expected_sources = set(item.get("expected_source_ids", []))
            
            # Lấy top-k kết quả thực tế do hệ thống trả về
            retrieved_sources = system_retrieval_results.get(question_id, [])[:k]
            retrieved_sources_set = set(retrieved_sources)
            
            # Nếu có ít nhất một nguồn đúng nằm trong top-k kết quả truy xuất -> Tính là 1 hit
            if expected_sources.intersection(retrieved_sources_set):
                hits += 1

        recall_score = hits / total_answerable if total_answerable > 0 else 0.0
        return recall_score

    def run_benchmark_report(self, system_retrieval_results: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Chạy tổng hợp báo cáo đánh giá hiệu năng retrieval ở các mức k khác nhau.
        """
        recall_at_1 = self.evaluate_retrieval_recall(system_retrieval_results, k=1)
        recall_at_3 = self.evaluate_retrieval_recall(system_retrieval_results, k=3)
        recall_at_5 = self.evaluate_retrieval_recall(system_retrieval_results, k=5)

        report = {
            "total_benchmark_questions": len(self.evaluation_dataset),
            "recall_at_1": round(recall_at_1 * 100, 2),
            "recall_at_3": round(recall_at_3 * 100, 2),
            "recall_at_5": round(recall_at_5 * 100, 2)
        }
        
        print("===== BÁO CÁO ĐÁNH GIÁ RETRIEVAL BENCHMARK =====")
        for metric, value in report.items():
            print(f"- {metric}: {value}%" if "recall" in metric else f"- {metric}: {value}")
        print("=================================================")
        
        return report


if __name__ == "__main__":
    # Đoạn code kiểm thử chạy độc lập module Evaluator với dữ liệu giả lập
    evaluator = RAGEvaluator()
    
    # Dữ liệu giả lập kết quả trả về từ hệ thống của bạn để kiểm thử
    mock_system_results = {
        "Q001": ["policy-document-01", "other-doc-99"],
        "Q002": ["hr-policy-03", "benefits-01"],
        "Q003": [] # Câu hỏi unanswerable
    }
    
    evaluator.run_benchmark_report(mock_system_results)