from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

all_docs = []
for path in Path("pdfs").glob("*.pdf"):
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    # optionally add source metadata
    for d in docs:
        d.metadata["source_file"] = str(path.name)
    all_docs.extend(docs)

print(f"Loaded {len(all_docs)} pages/chunks from PDFs.")
