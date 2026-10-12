"""
apps/web/app.py
Giao diện người dùng trực quan bằng Streamlit cho Enterprise Knowledge AI Assistant,
hỗ trợ khung chat hỏi đáp, hiển thị trích dẫn nguồn và phân quyền truy cập.
"""

import streamlit as st

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
        if "citations" in message and message["citations"]:
            with st.expander("📌 Xem tài liệu trích dẫn nguồn (Citations)"):
                for cit in message["citations"]:
                    st.code(f"Source ID: {cit}", language="text")

# Ô nhập câu hỏi từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn về tài liệu nội bộ..."):
    # Hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Xử lý sinh câu trả lời từ AI (Mô phỏng gọi API backend)
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            # Logic gọi backend FastAPI tương ứng ở đây
            ai_response = f"Đây là câu trả lời tổng hợp dựa trên kho tài liệu nội bộ phù hợp với quyền hạn **{user_role}** của bạn."
            simulated_citations = ["policy-document-01", "hr-policy-03"]
            
            st.markdown(ai_response)
            with st.expander("📌 Xem tài liệu trích dẫn nguồn (Citations)"):
                for cit in simulated_citations:
                    st.code(f"Source ID: {cit}", language="text")
                    
            # Lưu lịch sử assistant
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_response,
                "citations": simulated_citations
            })