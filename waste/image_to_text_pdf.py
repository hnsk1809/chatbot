from pdf2image import convert_from_path
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

poppler_path = r"C:\Users\hnsk9\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"  # point to your bin folder

pages = convert_from_path("pdfs/10_marksheet.pdf", dpi=300, poppler_path=poppler_path)
texts = []
for i, page_image in enumerate(pages):
    text = pytesseract.image_to_string(page_image, lang="eng")
    texts.append({"page": i+1, "text": text.strip()})


def clean_text(s):
    s = s.replace("\r", " ")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)  # collapse multiple spaces/newlines
    return s.strip()

texts = [ {"page": t["page"], "text": clean_text(t["text"])} for t in texts ]
for t in texts:
    print(f"Page {t['page']}: {t['text'][:]}")
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    for item in texts:
        f.write(f"=== PAGE {item['page']} ===\n")
        f.write(item["text"] + "\n\n")

