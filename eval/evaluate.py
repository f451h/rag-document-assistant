"""Retrieval evaluation on the sample handbook.

For each question we know which page holds the answer. A retriever "hits" if a chunk
from that page shows up in its top k results. Reports Hit@1, Hit@3, Hit@5 and MRR for
keyword-only, semantic-only and hybrid search.

    python -m eval.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

from rag import DocumentIndex
from rag.embeddings import Embedder

ROOT = Path(__file__).resolve().parents[1]


def evaluate(index: DocumentIndex, questions: list[dict], method: str, k: int = 5) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    rr = 0.0
    for q in questions:
        results = index.retriever.search(q["question"], k=k, method=method)
        pages = [h.chunk.page for h in results]
        rank = next((i + 1 for i, p in enumerate(pages) if p == q["page"]), None)
        if rank:
            rr += 1 / rank
            for cut in hits:
                hits[cut] += rank <= cut
    n = len(questions)
    return {"method": method, **{f"hit@{c}": hits[c] / n for c in hits}, "mrr": rr / n}


def main():
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())
    pdf = ROOT / "sample_docs" / "northwind_handbook.pdf"
    index = DocumentIndex.from_files([(pdf.name, pdf.read_bytes())], embedder=Embedder())
    methods = ["bm25"] + (["dense", "hybrid"] if index.retriever.vectors is not None else [])
    if index.retriever.vectors is None:
        print("Embedding model not available - evaluating keyword search only.")

    rows = [evaluate(index, questions, m) for m in methods]
    lines = ["| Method | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@5']:.2f} | {r['mrr']:.3f} |")
    table = "\n".join(lines)
    print(f"{len(questions)} questions, {len(index.chunks)} chunks\n")
    print(table)
    (ROOT / "eval" / "results.md").write_text(f"{len(questions)} questions on the sample handbook\n\n{table}\n")


if __name__ == "__main__":
    main()
