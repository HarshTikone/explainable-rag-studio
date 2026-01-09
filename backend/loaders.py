from typing import List, Dict, Any
from pypdf import PdfReader

def load_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Loads a PDF and returns a list of page dicts:
      {
        "source": filename,
        "page": page_number (1-based),
        "text": extracted_text
      }
    """
    reader = PdfReader(pdf_path)
    source = pdf_path.split("/")[-1].split("\\")[-1]
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.replace("\x00", " ").strip()
        if text:
            pages.append({"source": source, "page": i + 1, "text": text})
    return pages
