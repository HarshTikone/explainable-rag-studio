from typing import List, Dict, Any
from pypdf import PdfReader
from pathlib import Path

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


def load_document_pages(path: str) -> List[Dict[str, Any]]:
    """Load supported public-demo and uploaded document formats into page records."""
    suffix = Path(path).suffix.casefold()
    if suffix == ".pdf":
        return load_pdf_pages(path)
    if suffix in {".txt", ".md"}:
        text = Path(path).read_text(encoding="utf-8").replace("\x00", " ").strip()
        return [{"source": Path(path).name, "page": 1, "text": text}] if text else []
    raise ValueError(f"Unsupported document type: {suffix}")
