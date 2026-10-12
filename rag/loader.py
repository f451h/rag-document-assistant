"""Turn uploaded files into page-level text, keeping track of where each page came from."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    source: str   # file name
    page: int     # 1-based page number (or section number for non-PDF files)
    text: str


def _clean(text: str) -> str:
    text = text.replace("­", "")                 # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)            # words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(data: bytes, name: str) -> list[Page]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        text = _clean(p.extract_text() or "")
        if text:
            pages.append(Page(source=name, page=i, text=text))
    return pages


def load_text(data: bytes, name: str, chars_per_page: int = 3000) -> list[Page]:
    """Plain text / markdown has no pages, so split it into page-sized sections."""
    text = _clean(data.decode("utf-8", errors="ignore"))
    pages, start, n = [], 0, 1
    while start < len(text):
        end = min(len(text), start + chars_per_page)
        if end < len(text):                       # try not to cut a paragraph in half
            cut = text.rfind("\n\n", start, end)
            end = cut if cut > start + chars_per_page // 2 else end
        pages.append(Page(source=name, page=n, text=text[start:end].strip()))
        start, n = end, n + 1
    return [p for p in pages if p.text]


def load_docx(data: bytes, name: str) -> list[Page]:
    from docx import Document
    doc = Document(io.BytesIO(data))
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return load_text(text.encode(), name)


def load_file(data: bytes, name: str) -> list[Page]:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(data, name)
    if suffix == ".docx":
        return load_docx(data, name)
    if suffix in {".txt", ".md"}:
        return load_text(data, name)
    raise ValueError(f"Unsupported file type: {suffix}")
