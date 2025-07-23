import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import re
from tqdm import tqdm

def text_formatter(text: str) -> str:
    # Simple text cleanup (customize as needed)
    return ' '.join(text.split())

def extract_text_from_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc), total=doc.page_count):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_texts.append({
            "page_number": page_number,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": int(len(text)/4),
            "text": text
        })
    return pages_and_texts

def detect_logical_page_numbers(pdf_file: str) -> list:
    reader = PdfReader(pdf_file)
    logical_map = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        lines = text.strip().splitlines()
        candidates = lines[-2:] + lines[:2]  # check top and bottom lines
        page_num_found = None
        for line in candidates:
            match = re.search(r'(?<!\w)(Page\s*)?(\d{1,4})(?!\w)', line.strip(), re.IGNORECASE)
            if match:
                page_num_found = int(match.group(2))
                break
        logical_map.append({
            "pdf_index": i,
            "logical_page": page_num_found if page_num_found else None,
            "text": text
        })
    return logical_map 