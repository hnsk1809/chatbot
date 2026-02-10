import streamlit as st
import os

# Import backend functions
from rag_core import create_or_load_retriever, build_qa_pipeline

# -------------------------------
# Step 1: Configure Streamlit page
# -------------------------------
st.set_page_config(
    page_title="PDF Question Answering System",
    layout="wide"
)

# -------------------------------
# Step 2: Page heading
# -------------------------------
st.title("📄 PDF Question Answering System")
st.caption("OCR + RAG + FAISS + Mistral LLM")

# -------------------------------
# Step 3: Upload PDF (UI responsibility)
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a scanned PDF",
    type=["pdf"]
)

# -------------------------------
# Step 4: Save uploaded PDF locally
# -------------------------------
if uploaded_file:
    os.makedirs("pdfs", exist_ok=True)

    pdf_path = os.path.join("pdfs", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully")

    # -------------------------------
    # Step 5: Build backend pipeline
    # -------------------------------
    with st.spinner("Processing document..."):
        retriever = create_or_load_retriever(pdf_path)
        qa_chain = build_qa_pipeline(retriever)

    st.success("Document ready for questions")

    # -------------------------------
    # Step 6: Ask questions
    # -------------------------------
    question = st.text_input("Ask a question from the document")

    if question:
        with st.spinner("Generating answer..."):
            result = qa_chain.invoke(question)

        # -------------------------------
        # Step 7: Display answer
        # -------------------------------
        st.subheader("Answer")
        st.write(result)
