from pathlib import Path

import numpy as np
import pytest

from rag import DocumentIndex
from rag.chunker import chunk_pages, split_sentences
from rag.embeddings import EmbeddingUnavailable
from rag.llm import build_messages, cited_numbers
from rag.loader import Page, load_file
from rag.retriever import HybridRetriever, reciprocal_rank_fusion, tokenize

SAMPLE = Path(__file__).resolve().parents[1] / "sample_docs" / "northwind_handbook.pdf"


class FakeEmbedder:
    """Bag-of-words 'embeddings' so the dense path can be tested without downloading a model."""

    def __init__(self):
        self.vocab = {}

    def _vec(self, text):
        v = np.zeros(512, dtype=np.float32)
        for t in tokenize(text):
            v[hash(t) % 512] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class BrokenEmbedder:
    def embed_documents(self, texts):
        raise EmbeddingUnavailable("offline")


@pytest.fixture(scope="module")
def sample_index():
    return DocumentIndex.from_files([(SAMPLE.name, SAMPLE.read_bytes())], embedder=None, use_dense=False)


def test_pdf_loads_with_page_numbers():
    pages = load_file(SAMPLE.read_bytes(), SAMPLE.name)
    assert [p.page for p in pages] == [1, 2, 3, 4, 5, 6]
    assert "annual leave" in pages[1].text


def test_text_file_is_split_into_sections():
    text = ("Paragraph one. " * 150 + "\n\n") * 3
    pages = load_file(text.encode(), "notes.txt")
    assert len(pages) > 1 and all(p.source == "notes.txt" for p in pages)


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        load_file(b"x", "image.png")


def test_sentence_split_keeps_numbers():
    s = split_sentences("Leave is 1.67 days per month. Sick leave is 10 days.")
    assert s == ["Leave is 1.67 days per month.", "Sick leave is 10 days."]


def test_chunks_stay_inside_one_page_and_respect_size():
    pages = [Page("a.pdf", 1, "One sentence here. " * 80), Page("a.pdf", 2, "Second page text. " * 5)]
    chunks = chunk_pages(pages, max_chars=300)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 320 for c in chunks)
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert max(fused, key=fused.get) == 1          # ranked high by both lists
    assert fused[3] > fused[2] and fused[3] > fused[4]


def test_bm25_finds_the_right_page(sample_index):
    hits, relevant = sample_index.retrieve("How many sick leave days do I get?")
    assert relevant and hits[0].chunk.page == 2


def test_irrelevant_question_is_flagged(sample_index):
    hits, relevant = sample_index.retrieve("What is the capital of Australia?")
    assert not relevant


def test_hybrid_mode_with_embedder():
    index = DocumentIndex.from_files([(SAMPLE.name, SAMPLE.read_bytes())], embedder=FakeEmbedder())
    assert index.retriever.mode == "hybrid"
    hits = index.retriever.search("meal allowance abroad", k=3)
    assert hits[0].chunk.page == 4
    assert hits[0].dense is not None


def test_falls_back_to_bm25_when_embeddings_fail():
    chunks = chunk_pages([Page("a.txt", 1, "Remote work is allowed. Expenses need receipts.")])
    r = HybridRetriever(chunks, embedder=BrokenEmbedder())
    assert r.mode == "bm25" and r.dense_error


def test_prompt_numbers_passages(sample_index):
    hits, _ = sample_index.retrieve("expense deadline", k=3)
    msgs = build_messages("expense deadline", hits)
    assert msgs[0]["role"] == "system"
    assert "[1] (source: northwind_handbook.pdf, page" in msgs[-1]["content"]


def test_cited_numbers_ignores_invalid_references():
    assert cited_numbers("Yes [1]. Also [3][2] and [9].", n_hits=3) == [1, 2, 3]
