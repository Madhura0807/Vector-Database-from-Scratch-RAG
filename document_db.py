import threading
from typing import List, Tuple
from dataclasses import dataclass

from brute_force import BruteForce, VectorItem
from hnsw import HNSW
from distance import cosine


@dataclass
class DocItem:
    """Document chunk with embedding."""
    id: int
    title: str
    text: str
    emb: List[float]


class DocumentDB:
    """Document database with HNSW index for semantic search."""

    def __init__(self):
        self.store: dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()  # Fallback for small sets
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0  # Determined from first embedding

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        """Insert a document chunk with its embedding."""
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        """
        Semantic search for top-k similar chunks.
        Returns list of (distance, DocItem) pairs.
        """
        with self.mu:
            if not self.store:
                return []

            # Use brute force for small sets, HNSW for larger
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)

            results = []
            for dist, id_ in raw:
                if id_ in self.store and dist <= max_dist:
                    results.append((dist, self.store[id_]))
            return results

    def remove(self, id_: int) -> bool:
        """Remove a document chunk."""
        with self.mu:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> List[DocItem]:
        """Get all document chunks."""
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        """Get number of document chunks."""
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        """Get embedding dimension."""
        return self.dims


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    """
    Split text into overlapping chunks.
    Default: 250 words per chunk with 30-word overlap.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunk = ' '.join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
    return chunks
