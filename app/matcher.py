from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .chunker import Chunk
from .utils import logger


@dataclass
class Clause:
    text: str
    score: float
    source: Chunk


class ClauseMatcher:
    """Extract clause-like sentences from retrieved chunks and score them.

    Scoring combines retrieval score and simple rule-based boosts
    for common compliance/insurance markers (dates, exclusions, waiting periods).
    """

    date_re = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
                         re.IGNORECASE)
    exclusion_re = re.compile(r"\b(exclusion|excluded|does not cover|not covered)\b", re.IGNORECASE)
    waiting_re = re.compile(r"\b(waiting period|after\s+\d+\s+days|effective after)\b", re.IGNORECASE)

    def select_clauses(self, question: str, retrieved: List[Chunk], max_per_chunk: int = 3) -> List[Chunk]:
        candidates: List[Chunk] = []
        for ch in retrieved:
            # split into sentences (simple heuristic)
            sentences = re.split(r"(?<=[.!?])\s+", ch.text)
            # keep mid-length sentences
            scored: List[tuple[str, float]] = []
            for s in sentences:
                s_clean = s.strip()
                if not s_clean or len(s_clean) < 20:
                    continue
                score = ch.score
                if self.date_re.search(s_clean):
                    score += 0.05
                if self.exclusion_re.search(s_clean):
                    score += 0.08
                if self.waiting_re.search(s_clean):
                    score += 0.05
                # prefer sentences that mention question keywords
                q_tokens = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 3]
                hits = sum(1 for t in q_tokens if t in s_clean.lower())
                score += min(0.1, 0.02 * hits)
                scored.append((s_clean, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            for s, sc in scored[:max_per_chunk]:
                # create a clause as a small Chunk for downstream compatibility
                clause = Chunk(text=s, meta=ch.meta)
                clause.score = float(sc)
                candidates.append(clause)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
