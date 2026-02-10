# langchain_qa_local.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1️⃣ Load PDF
loader = PyPDFLoader("pdfs/coa.pdf")
docs = loader.load()

# 2️⃣ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(docs)

# 3️⃣ Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4️⃣ Build FAISS vectorstore
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("faiss_index")

# 5️⃣ Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 6️⃣ Load local HuggingFace model (distilgpt2)
model_name = "distilgpt2"  # lightweight CPU-friendly model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,  # generates 150 new tokens
    do_sample=True,
    temperature=0.7,
    device_map="auto"
)

llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={"max_new_tokens": 150, "do_sample": True, "temperature": 0.7, "pad_token_id": tokenizer.eos_token_id}
)

# 7️⃣ Build RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 8️⃣ Ask a question
query = "what are logic micro operations?"
result = qa.invoke(query)

print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.metadata['page'])
