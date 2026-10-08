import threading
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass

from brute_force import BruteForce, VectorItem
from kd_tree import KDTree
from hnsw import HNSW
from distance import get_dist_fn, DistFn


@dataclass
class Hit:
    """Search result hit."""
    id: int
    meta: str
    cat: str
    emb: List[float]
    dist: float


@dataclass
class SearchOut:
    """Search output with results and timing."""
    hits: List[Hit]
    us: int
    algo: str
    metric: str


@dataclass
class BenchOut:
    """Benchmark output comparing all algorithms."""
    bf_us: int
    kd_us: int
    hnsw_us: int
    n: int


class VectorDB:
    """Unified vector database with three search algorithms."""

    def __init__(self, dims: int):
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = dims

    def insert(self, meta: str, cat: str, emb: List[float], dist_fn: DistFn) -> int:
        """Insert a vector into all three indexes."""
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist_fn)
            return v.id

    def remove(self, id_: int) -> bool:
        """Remove a vector from all indexes."""
        with self.mu:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            # Rebuild KD-Tree (it doesn't support efficient deletion)
            remaining = list(self.store.values())
            self.kdt.rebuild(remaining)
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> SearchOut:
        """
        Search for k-nearest neighbors.
        Returns SearchOut with hits and timing.
        """
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:  # hnsw
                raw = self.hnsw.knn(q, k, 50, dfn)

            us = int((time.perf_counter() - t0) * 1_000_000)

            out = SearchOut(hits=[], us=us, algo=algo, metric=metric)
            for dist, id_ in raw:
                if id_ in self.store:
                    out.hits.append(Hit(
                        id=id_,
                        meta=self.store[id_].metadata,
                        cat=self.store[id_].category,
                        emb=self.store[id_].emb,
                        dist=dist
                    ))
            return out

    def benchmark(self, q: List[float], k: int, metric: str) -> BenchOut:
        """Benchmark all three algorithms on the same query."""
        with self.mu:
            dfn = get_dist_fn(metric)

            def time_fn(fn):
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            bf_us = time_fn(lambda: self.bf.knn(q, k, dfn))
            kd_us = time_fn(lambda: self.kdt.knn(q, k, dfn))
            hnsw_us = time_fn(lambda: self.hnsw.knn(q, k, 50, dfn))

            return BenchOut(bf_us=bf_us, kd_us=kd_us, hnsw_us=hnsw_us, n=len(self.store))

    def all(self) -> List[VectorItem]:
        """Get all stored vectors."""
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        """Get HNSW graph information."""
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        """Get number of stored vectors."""
        with self.mu:
            return len(self.store)
