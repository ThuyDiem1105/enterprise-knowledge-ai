# ==========================================
# FILE: app.py
# Chức năng: Giao diện Streamlit tích hợp gọi Generator (Ollama local) thực tế
# ==========================================

import streamlit as st
import sys
import os

# Thêm thư mục gốc vào sys.path để Python nhận diện được package 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../")) if "apps" in current_dir else current_dir
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import chuẩn từ thư mục src.generation
try:
    from src.generation.generator import generate_rag_answer
except ImportError:
    try:
        from generator import generate_rag_answer
    except ImportError:
        generate_rag_answer = None
st.set_page_config(
    page_title="Enterprise Knowledge AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Sidebar: Cấu hình và Phân quyền ---
st.sidebar.title("⚙️ Cấu hình Hệ thống")
user_role = st.sidebar.selectbox(
    "Vai trò người dùng (Access Scope):",
    ["public", "internal", "confidential"],
    index=1
)
model_choice = st.sidebar.selectbox("Chọn mô hình LLM", ["llama3:8b"])

st.sidebar.markdown("---")
st.sidebar.info(
    "Hệ thống RAG doanh nghiệp hỗ trợ tìm kiếm tài liệu nội bộ, "
    "trích xuất bằng chứng chuẩn xác và chống bịa đặt dữ liệu."
)

# --- Giao diện chính ---
st.title("📚 Enterprise Knowledge AI Assistant")
st.markdown("Hệ thống AI giúp tìm kiếm, truy xuất và trả lời câu hỏi dựa trên kho tài liệu nội bộ doanh nghiệp.")

# Khởi tạo lịch sử chat trong session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chunks" in message and message["chunks"]:
            with st.expander("📌 Xem các đoạn tài liệu nguồn (Chunks)"):
                for c in message["chunks"]:
                    st.markdown(f"**Nguồn ID: `{c.get('document_id')}`**")
                    st.text(c.get('content'))

# Ô nhập câu hỏi từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn về tài liệu nội bộ..."):
    # Hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dữ liệu chunks mô phỏng dựa theo kho tài liệu thực tế của bạn
    # (Sau này khi hoàn thiện Giai đoạn 4, phần này sẽ được thay bằng kết quả từ Retrieval Engine)
    retrieved_chunks = [
        {
            "document_id": "policy-document-01",
            "content": "Theo quy định của Luật an toàn thông tin mạng 2015, an toàn thông tin mạng là việc bảo vệ và phòng ngừa các nguy cơ tấn công, xâm phạm hoặc phá hủy hệ thống thông tin mạng để đảm bảo tính liên tục, an toàn và hiệu quả trong hoạt động cung cấp dịch vụ thông tin trên mạng."
        }
    ]

    # Xử lý sinh câu trả lời thực tế từ AI (Ollama local)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất tài liệu và tổng hợp câu trả lời từ Ollama..."):
            if generate_rag_answer:
                # Gọi hàm generator thật đã tích hợp Ollama
                ai_response = generate_rag_answer(prompt, retrieved_chunks)
            else:
                ai_response = "Lỗi: Không tìm thấy module generator.py. Hãy kiểm tra lại đường dẫn."
            
            st.markdown(ai_response)
            
            # Hiển thị trích dẫn nguồn
            with st.expander("📌 Xem các đoạn tài liệu nguồn (Chunks)"):
                for c in retrieved_chunks:
                    st.markdown(f"**Nguồn ID: `{c.get('document_id')}`**")
                    st.text(c.get('content'))
                    
            # Lưu lịch sử assistant
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_response,
                "chunks": retrieved_chunks
            })