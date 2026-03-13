from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector.copy()
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


@dataclass
class Neighbor:
    word: str
    score: float


class GloveIndex:
    """
    In-memory GloVe index.

    We keep normalized vectors so cosine similarity is just a dot product.
    """

    def __init__(self, words: list[str], matrix: np.ndarray) -> None:
        self.words = words
        self.matrix = matrix
        self.lookup = {word: idx for idx, word in enumerate(words)}

    @classmethod
    def load(cls, path: Path) -> "GloveIndex":
        words: list[str] = []
        rows: list[np.ndarray] = []

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.strip().split()
                if len(parts) <= 2:
                    continue
                word = parts[0]
                try:
                    vector = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
                except ValueError as exc:
                    raise ValueError(f"failed to parse vector on line {line_number}") from exc
                words.append(word)
                rows.append(normalize(vector))

        if not rows:
            raise RuntimeError(f"no GloVe vectors loaded from {path}")

        matrix = np.vstack(rows)
        return cls(words, matrix)

    @property
    def dims(self) -> int:
        return int(self.matrix.shape[1])

    def vector(self, word: str) -> np.ndarray:
        key = word.lower()
        if key not in self.lookup:
            raise KeyError(f"{word!r} not found in the vocabulary")
        return self.matrix[self.lookup[key]]

    def neighbors(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        exclude: Iterable[str] | None = None,
    ) -> list[Neighbor]:
        excluded = {value.lower() for value in (exclude or [])}
        scores = self.matrix @ normalize(query_vector)
        ranked = np.argsort(-scores)
        result: list[Neighbor] = []
        for idx in ranked:
            word = self.words[int(idx)]
            if word.lower() in excluded:
                continue
            result.append(Neighbor(word=word, score=float(scores[int(idx)])))
            if len(result) >= top_k:
                break
        return result


def analogy_vector(index: GloveIndex, positive: list[str], negative: list[str]) -> np.ndarray:
    if not positive:
        raise ValueError("at least one positive term is required")
    vector = np.zeros(index.dims, dtype=np.float32)
    for word in positive:
        vector += index.vector(word)
    for word in negative:
        vector -= index.vector(word)
    return normalize(vector)


def project_local_neighborhood(query_vector: np.ndarray, neighbor_vectors: list[np.ndarray]) -> np.ndarray:
    """
    Lightweight PCA with NumPy only.

    This keeps dependencies small while still giving students a meaningful local view.
    """
    stacked = np.vstack([query_vector, *neighbor_vectors])
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return centered @ basis


def save_plot(
    query_label: str,
    query_vector: np.ndarray,
    neighbors: list[Neighbor],
    index: GloveIndex,
    output_path: Path,
) -> None:
    neighbor_vectors = [index.vector(item.word) for item in neighbors]
    projected = project_local_neighborhood(query_vector, neighbor_vectors)

    plt.figure(figsize=(8, 6))
    for idx, (x, y) in enumerate(projected):
        if idx == 0:
            label = query_label
            color = "#d97706"
            size = 90
        else:
            label = neighbors[idx - 1].word
            color = "#118a61"
            size = 48
        plt.scatter([x], [y], c=color, s=size)
        plt.text(x + 0.02, y + 0.02, label, fontsize=10)

    plt.title(f"Local neighborhood for {query_label}")
    plt.xlabel("principal direction 1")
    plt.ylabel("principal direction 2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
