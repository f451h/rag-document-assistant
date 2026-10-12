"""Hybrid retrieval: BM25 keyword search + dense embeddings, merged with Reciprocal Rank Fusion.

Why both: BM25 is great at exact terms (policy names, numbers, acronyms) while embeddings
catch paraphrases ("time off" vs "annual leave"). RRF combines the two rankings without
having to put BM25 scores and cosine similarities on the same scale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .embeddings import Embedder, EmbeddingUnavailable

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its many much
of on or our so that the their them there these they this to was we what when where which who
why will with you your my me am""".split())


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Each ranking is a list of chunk ids, best first. Returns id -> fused score."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float = 0.0
    dense: float | None = None
    sources: list[str] = field(default_factory=list)   # which retrievers found it


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, use_dense: bool = True):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.embedder = embedder
        self.vectors: np.ndarray | None = None
        self.dense_error: str | None = None
        if use_dense and embedder is not None and chunks:
            try:
                self.vectors = embedder.embed_documents([c.text for c in chunks])
            except EmbeddingUnavailable as e:
                self.dense_error = str(e)

    @property
    def mode(self) -> str:
        return "hybrid" if self.vectors is not None else "bm25"

    def search(self, query: str, k: int = 5, method: str | None = None, candidates: int = 20) -> list[Hit]:
        method = method or self.mode
        if not self.chunks:
            return []
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = [i for i in np.argsort(-bm25_scores) if bm25_scores[i] > 0][:candidates]

        dense_scores = None
        dense_rank: list[int] = []
        if method in ("hybrid", "dense") and self.vectors is not None:
            q = self.embedder.embed_query(query)
            dense_scores = self.vectors @ q
            dense_rank = list(np.argsort(-dense_scores)[:candidates])

        if method == "bm25" or dense_scores is None:
            ranking, fused = bm25_rank, {i: float(bm25_scores[i]) for i in bm25_rank}
        elif method == "dense":
            ranking, fused = dense_rank, {i: float(dense_scores[i]) for i in dense_rank}
        else:
            fused = reciprocal_rank_fusion([bm25_rank, dense_rank])
            ranking = sorted(fused, key=fused.get, reverse=True)

        hits = []
        for i in ranking[:k]:
            found_by = [name for name, r in (("keyword", bm25_rank), ("semantic", dense_rank)) if i in r]
            hits.append(Hit(chunk=self.chunks[i], score=fused[i], bm25=float(bm25_scores[i]),
                            dense=None if dense_scores is None else float(dense_scores[i]), sources=found_by))
        return hits

    def is_relevant(self, hits: list[Hit], min_bm25: float = 1.0, min_cosine: float = 0.55) -> bool:
        """Cheap guard so we don't ask the LLM to answer from unrelated text."""
        if not hits:
            return False
        top = hits[0]
        if top.dense is not None and max(h.dense for h in hits) >= min_cosine:
            return True
        return max(h.bm25 for h in hits) >= min_bm25
