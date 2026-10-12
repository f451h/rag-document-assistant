"""Split pages into overlapping chunks along sentence boundaries."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Page

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    id: int
    source: str
    page: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}, p. {self.page}"


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if para:
            parts.extend(s.strip() for s in _SENTENCE_END.split(para) if s.strip())
    return parts


def chunk_pages(pages: list[Page], max_chars: int = 700, overlap_sentences: int = 1) -> list[Chunk]:
    """Group sentences into chunks of up to ~max_chars, never crossing a page boundary.

    Keeping chunks inside one page means every chunk has exactly one page number, so
    citations always point to the right place. The last sentence of each chunk is repeated
    at the start of the next one, so an answer that spans two chunks isn't lost.
    """
    chunks: list[Chunk] = []
    for page in pages:
        sentences = split_sentences(page.text)
        current: list[str] = []
        for sent in sentences:
            if current and len(" ".join(current + [sent])) > max_chars:
                chunks.append(Chunk(len(chunks), page.source, page.page, " ".join(current)))
                current = current[-overlap_sentences:] if overlap_sentences else []
            current.append(sent)
        if current:
            text = " ".join(current)
            # skip a trailing chunk that only repeats the overlap sentence
            if not chunks or chunks[-1].page != page.page or text not in chunks[-1].text:
                chunks.append(Chunk(len(chunks), page.source, page.page, text))
    return chunks
