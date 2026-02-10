# image_n_text_pdf.py
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langdetect import detect
from pdf2image import convert_from_path
import pytesseract

# --- 1️⃣ Load PDF ---
pdf_path = "pdfs/10_marksheet.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# --- 2️⃣ Check if PDF has text; else run OCR ---
def extract_text_from_image_pdf(pdf_path):
    text_pages = []
    images = convert_from_path(pdf_path)
    for i, img in enumerate(images):
        txt = pytesseract.image_to_string(img)
        text_pages.append(txt)
    return text_pages

# Detect if pages are empty
empty_pages = all(len(d.page_content.strip()) == 0 for d in docs)
if empty_pages:
    print("No text found in PDF, running OCR...")
    ocr_texts = extract_text_from_image_pdf(pdf_path)
    docs = []
    for i, page_text in enumerate(ocr_texts):
        docs.append(
            type("Document", (object,), {"page_content": page_text, "metadata": {"page": i + 1}})()
        )

# --- 3️⃣ Keep only English lines ---
for doc in docs:
    lines = doc.page_content.split("\n")
    english_lines = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                if detect(line) == "en":
                    english_lines.append(line)
            except:
                continue
    doc.page_content = "\n".join(english_lines)

# --- 4️⃣ Split into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(docs)

# --- 5️⃣ Create embeddings and vectorstore ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("faiss_index")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 6️⃣ Load local HuggingFace model ---
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    device_map="auto"
)

llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={"max_new_tokens": 150, "do_sample": True, "temperature": 0.7, "pad_token_id": tokenizer.eos_token_id}
)

# --- 7️⃣ Build RetrievalQA chain ---
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- 8️⃣ Ask a question ---
query = "score in mathematics"
result = qa.invoke(query)

# --- 9️⃣ Extract numeric score ---
def extract_numeric_score(text, subject="Mathematics"):
    for line in text.split("\n"):
        if subject.lower() in line.lower():
            match = re.search(r"(\d+)", line)
            if match:
                return int(match.group(1))
    return None

# Go through retrieved source documents to find numeric score
numeric_score = None
for doc in result["source_documents"]:
    numeric_score = extract_numeric_score(doc.page_content, subject="Mathematics")
    if numeric_score is not None:
        break

# --- 10️⃣ Print results ---
print("Answer (text):", result["result"])
print("Mathematics Score (number):", numeric_score)
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.metadata['page'])
