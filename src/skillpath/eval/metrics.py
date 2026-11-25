"""Ranking metrics for the offline evaluation.

All functions take `ranked`, the recommended item ids in rank order, and `relevance`, a
dict from item id to graded relevance (0 = not relevant). Items missing from `relevance`
count as 0. An item is binary-relevant when its grade is at least `threshold`.
"""

import math


def _relevant(relevance: dict[str, float], threshold: float) -> set[str]:
    return {i for i, g in relevance.items() if g >= threshold}


def precision_at_k(ranked: list[str], relevance: dict[str, float], k: int, threshold: float = 1) -> float:
    rel = _relevant(relevance, threshold)
    return sum(i in rel for i in ranked[:k]) / k if k else 0.0


def recall_at_k(ranked: list[str], relevance: dict[str, float], k: int, threshold: float = 1) -> float:
    rel = _relevant(relevance, threshold)
    return sum(i in rel for i in ranked[:k]) / len(rel) if rel else 0.0


def f1_at_k(ranked: list[str], relevance: dict[str, float], k: int, threshold: float = 1) -> float:
    p, r = precision_at_k(ranked, relevance, k, threshold), recall_at_k(ranked, relevance, k, threshold)
    return 2 * p * r / (p + r) if p + r else 0.0


def ndcg_at_k(ranked: list[str], relevance: dict[str, float], k: int) -> float:
    """NDCG with graded gains (2^grade - 1). The ideal ranking comes from all judged items."""
    def dcg(grades):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))
    actual = dcg([relevance.get(i, 0) for i in ranked[:k]])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0
