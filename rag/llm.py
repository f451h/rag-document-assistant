"""Answer generation with Groq (Llama 3.3 70B by default)."""
from __future__ import annotations

import os
import re
from typing import Iterator

from .retriever import Hit

DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages you are given.

Rules:
- Every factual sentence must end with the passage number(s) it came from, like [1] or [2][3].
- If the passages don't contain the answer, say exactly: "I couldn't find that in the uploaded documents."
  Do not use outside knowledge to fill gaps.
- Quote numbers, dates and amounts exactly as written.
- Keep answers short and direct. Use a short bullet list only when listing several items."""


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for n, h in enumerate(hits, start=1):
        blocks.append(f"[{n}] (source: {h.chunk.source}, page {h.chunk.page})\n{h.chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:          # a little memory for follow-up questions
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"})
    return messages


def cited_numbers(answer: str, n_hits: int) -> list[int]:
    """Citation numbers that actually appear in the answer and point to a real passage."""
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in nums if 1 <= n <= n_hits)


class GroqLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None, temperature: float = 0.1):
        from groq import Groq
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to a .env file or paste it in the sidebar.")
        self.client = Groq(api_key=key)
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.temperature = temperature

    def stream(self, messages: list[dict]) -> Iterator[str]:
        resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                   temperature=self.temperature, stream=True)
        for event in resp:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    def complete(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                   temperature=self.temperature)
        return resp.choices[0].message.content
