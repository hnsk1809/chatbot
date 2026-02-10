# extract_pdfplumber.py
import pdfplumber

full_text = []
with pdfplumber.open("coa.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            full_text.append({"page": i+1, "text": text.strip()})
        else:
            full_text.append({"page": i+1, "text": ""})

print(full_text[1]["text"][:1000])
