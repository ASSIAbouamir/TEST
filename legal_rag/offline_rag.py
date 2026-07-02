"""
Offline RAG utilities for this project.

Why:
- Network access is blocked in this environment, so we cannot install or call
  dependencies like llama_index / groq / langgraph / sentence_transformers.
- The processed documents contain injected "[Rappel ...]" blocks that must be
  ignored to regenerate answers "from scratch" from the legal text itself.

This module provides:
- Cleaning helpers (strip [Rappel ...] blocks, skip placeholders)
- A tiny BM25 implementation (no external deps, uses only stdlib)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .models import DocumentNode


_RE_RAPPEL_BLOCK = re.compile(r"\[Rappel.*?\]", flags=re.S)
_RE_TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", flags=re.UNICODE)


def strip_rappel_blocks(text: str) -> str:
    return _RE_RAPPEL_BLOCK.sub("", text or "").strip()


def is_placeholder_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if "image_url_placeholder" in t:
        return True
    if t.startswith("![") and t.endswith(")"):
        return True
    return False


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    accents_and_corruptions = ["", "é", "è", "ê", "ë", "à", "â", "ä", "ô", "ö", "û", "ù", "ü", "î", "ï", "ç"]
    for char in accents_and_corruptions:
        text = text.replace(char, "")
    return _RE_TOKEN.findall(text)


@dataclass(frozen=True)
class Retrieved:
    node: DocumentNode
    score: float
    cleaned_text: str


class BM25Index:
    """
    Minimal BM25 implementation (Okapi BM25-like).
    Good enough for ~1k nodes; keeps code dependency-free.
    """

    def __init__(self, documents: List[Tuple[DocumentNode, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)

        self.nodes: List[DocumentNode] = []
        self.texts: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.doc_len: List[int] = []

        df: Dict[str, int] = {}

        for node, cleaned_text in documents:
            self.nodes.append(node)
            self.texts.append(cleaned_text)
            tokens = tokenize(cleaned_text)
            self.doc_tokens.append(tokens)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            dl = len(tokens)
            self.doc_len.append(dl)
            for term in freqs.keys():
                df[term] = df.get(term, 0) + 1

        self.N = len(self.nodes)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        # IDF with +1 to keep it positive and stable for small corpora.
        self.idf: Dict[str, float] = {}
        for term, n_qi in df.items():
            self.idf[term] = math.log(((self.N - n_qi + 0.5) / (n_qi + 0.5)) + 1.0)

    def retrieve(self, query: str, top_k: int = 12) -> List[Retrieved]:
        if not self.N:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, freqs in enumerate(self.doc_freqs):
            score = 0.0
            dl = self.doc_len[idx]
            norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
            for term in q_tokens:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + norm
                score += idf * (tf * (self.k1 + 1.0)) / (denom if denom else 1.0)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        out: List[Retrieved] = []
        for idx, score in scores[: max(1, int(top_k))]:
            out.append(Retrieved(node=self.nodes[idx], score=float(score), cleaned_text=self.texts[idx]))
        return out


def build_clean_corpus(nodes: Iterable[DocumentNode]) -> List[Tuple[DocumentNode, str]]:
    corpus: List[Tuple[DocumentNode, str]] = []
    for node in nodes:
        cleaned = strip_rappel_blocks(node.content)
        if is_placeholder_text(cleaned):
            continue
        corpus.append((node, cleaned))
    return corpus


def extract_article_legal_sentence(text: str, article_num: int) -> Optional[str]:
    """
    Try to extract the legal sentence(s) for an article while dropping obvious
    injected commentary that starts with patterns like "L'Article ..." / "Au Benin ...".
    """
    cleaned = strip_rappel_blocks(text)
    if not cleaned:
        return None

    # Find the first occurrence of "Article X"
    m = re.search(rf"\bArticle\s+{int(article_num)}\b\s*:?\s*", cleaned, flags=re.I)
    if not m:
        return None

    tail = cleaned[m.start():].strip()

    # Split into sentences (roughly). Keep until a commentary marker appears.
    parts = [p.strip() for p in re.split(r"(?<=\.)\s+", tail) if p.strip()]
    kept: List[str] = []
    for p in parts:
        low = p.lower()
        if kept and (low.startswith("l'article") or low.startswith("au bénin") or low.startswith("au benin")):
            break
        kept.append(p)
        # Often the article in this corpus is 1 sentence; avoid swallowing too much.
        if len(kept) >= 3:
            break

    return " ".join(kept).strip() if kept else None

