"""Shared metric calculations for context engineering benchmarks."""

from __future__ import annotations

import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken.

    Falls back to cl100k_base encoding for unknown models.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def precision(true_positives: int, false_positives: int) -> float:
    """Calculate precision: TP / (TP + FP)."""
    total = true_positives + false_positives
    if total == 0:
        return 0.0
    return true_positives / total


def recall(true_positives: int, false_negatives: int) -> float:
    """Calculate recall: TP / (TP + FN)."""
    total = true_positives + false_negatives
    if total == 0:
        return 0.0
    return true_positives / total


def f1_score(prec: float, rec: float) -> float:
    """Calculate F1 score: harmonic mean of precision and recall."""
    total = prec + rec
    if total == 0.0:
        return 0.0
    return 2 * (prec * rec) / total


def compression_ratio(original_tokens: int, compressed_tokens: int) -> float:
    """Calculate compression ratio: compressed / original.

    Lower is better compression. Returns 1.0 if original is 0.
    """
    if original_tokens == 0:
        return 1.0
    return compressed_tokens / original_tokens


def effective_token_ratio(
    total_tokens: int,
    contributing_tokens: int,
) -> float:
    """Calculate what fraction of tokens contributed to the correct answer.

    Higher is better. Returns 0.0 if total is 0.
    """
    if total_tokens == 0:
        return 0.0
    return contributing_tokens / total_tokens


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Calculate Jaccard similarity between two sets of strings."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def cosine_similarity_simple(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors without numpy dependency at import time."""
    import numpy as np

    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
