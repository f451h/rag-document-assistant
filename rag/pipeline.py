"""Glue code: files in, retriever out, answers back."""
from __future__ import annotations

from dataclasses import dataclass, field

from .chunker import Chunk, chunk_pages
from .embeddings import Embedder
from .loader import Page, load_file
from .retriever import Hit, HybridRetriever

NOT_FOUND = "I couldn't find that in the uploaded documents."


@dataclass
class DocumentIndex:
    pages: list[Page] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    retriever: HybridRetriever | None = None

    @classmethod
    def from_files(cls, files: list[tuple[str, bytes]], embedder: Embedder | None = None,
                   max_chars: int = 450, use_dense: bool = True) -> "DocumentIndex":
        pages: list[Page] = []
        for name, data in files:
            pages.extend(load_file(data, name))
        chunks = chunk_pages(pages, max_chars=max_chars)
        retriever = HybridRetriever(chunks, embedder=embedder, use_dense=use_dense)
        return cls(pages=pages, chunks=chunks, retriever=retriever)

    @property
    def sources(self) -> list[str]:
        return sorted({p.source for p in self.pages})

    def retrieve(self, question: str, k: int = 5) -> tuple[list[Hit], bool]:
        hits = self.retriever.search(question, k=k)
        return hits, self.retriever.is_relevant(hits)
