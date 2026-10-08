from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class VectorItem:
    """Vector item with metadata."""
    id: int
    metadata: str
    category: str
    emb: List[float]


class BruteForce:
    """Brute force k-nearest neighbor search."""

    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        """Insert a vector item."""
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist_fn) -> List[Tuple[float, int]]:
        """
        Find k-nearest neighbors using brute force.
        Returns list of (distance, id) pairs sorted by distance.
        """
        results = []
        for v in self.items:
            results.append((dist_fn(q, v.emb), v.id))
        results.sort(key=lambda x: x[0])
        if len(results) > k:
            results = results[:k]
        return results

    def remove(self, id: int) -> None:
        """Remove item by id."""
        self.items = [v for v in self.items if v.id != id]
