# extract_pypdf2.py
from PyPDF2 import PdfReader

reader = PdfReader("coa.pdf")
full_text = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        text = text.strip()
        full_text.append({"page": i+1, "text": text})
    else:
        full_text.append({"page": i+1, "text": ""})

# print page 1 text
print(full_text[0]["text"][:500])  # show first 500 chars
