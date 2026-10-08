import math
from typing import Callable, List

# Distance function type
DistFn = Callable[[List[float], List[float]], float]


def euclidean(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)


def cosine(a: List[float], b: List[float]) -> float:
    """Calculate cosine distance (1 - cosine similarity) between two vectors."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: List[float], b: List[float]) -> float:
    """Calculate Manhattan distance between two vectors."""
    s = 0.0
    for i in range(len(a)):
        s += abs(a[i] - b[i])
    return s


def get_dist_fn(metric: str) -> DistFn:
    """Get distance function by metric name."""
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean
