import os
import requests
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# Use Chroma instead of FAISS for Windows-friendly vectorstore
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    try:
        from langchain.vectorstores import Chroma
    except Exception:
        Chroma = None
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM as LLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import Generation, LLMResult
from typing import Optional, List, Any
import json

# -------------------------------
# Step 1: OCR PDF (image -> text)
# -------------------------------
def pdf_to_text(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text


# -------------------------------
# Step 2: Split text into chunks
# -------------------------------
def split_text(text: str, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# -------------------------------
# Step 3: Vectorstore folder per PDF
# -------------------------------
def get_vectorstore_path(pdf_path: str) -> str:
    folder = "faiss_index"
    os.makedirs(folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(folder, base_name)


# -------------------------------
# Step 4: Create or load FAISS vectorstore
# -------------------------------
def create_or_load_retriever(pdf_path: str):
    """Create or load a Chroma vectorstore and return a LangChain retriever."""
    vectorstore_path = get_vectorstore_path(pdf_path)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Prefer Chroma (works well on Windows). If unavailable, raise helpful error.
    if Chroma is None:
        raise ImportError(
            "Chroma vectorstore is not available. Install `chromadb` and a compatible langchain vectorstore wrapper."
        )

    # Use a persistent directory per PDF
    persist_directory = vectorstore_path

    # If already persisted, load from disk
    if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        pdf_text = pdf_to_text(pdf_path)
        chunks = split_text(pdf_text)

        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        try:
            vectorstore.persist()
        except Exception:
            # Some Chroma wrappers persist automatically; ignore if not implemented
            pass

    return vectorstore.as_retriever(search_kwargs={"k": 5})


# -------------------------------
# Step 5: LangChain-compatible Mistral LLM
# -------------------------------
class MistralLLM(LLM):
    api_key: str
    model: str = "mistral-large-latest"
    temperature: float = 0.3
    max_tokens: int = 200

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            gen = Generation(text=text)
            generations.append([gen])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "mistral_api"


# -------------------------------
# Step 6: Build RAG chain (modern pattern without RetrievalQA)
# -------------------------------
def build_qa_pipeline(retriever):
    """Build a RAG pipeline using modern langchain pattern with runnables"""
    
    # Read API key securely from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")

    # Create the Mistral LLM
    mistral_llm = MistralLLM(api_key=api_key)

    # Define the prompt template
    prompt_template_str = """You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template_str
    )

    # Build the RAG chain using runnables (modern pattern)
    # Format documents from retriever
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the chain
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | mistral_llm
        | StrOutputParser()
    )

    return rag_chain
