import heapq
from typing import List, Tuple, Optional
from brute_force import VectorItem


class KDNode:
    """KD-Tree node."""
    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional['KDNode'] = None
        self.right: Optional['KDNode'] = None


class KDTree:
    """K-Dimensional Tree for exact nearest neighbor search."""

    def __init__(self, dims: int):
        self.root: Optional[KDNode] = None
        self.dims = dims

    def _destroy(self, node: Optional[KDNode]) -> None:
        """Recursively destroy tree nodes."""
        if node is None:
            return
        self._destroy(node.left)
        self._destroy(node.right)
        # Python garbage collection handles memory

    def _insert(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        """Recursively insert a vector item."""
        if node is None:
            return KDNode(v)
        axis = depth % self.dims
        if v.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def _knn(self, node: Optional[KDNode], q: List[float], k: int, depth: int,
             dist_fn, heap: List[Tuple[float, int]]) -> None:
        """Recursively search for k-nearest neighbors."""
        if node is None:
            return

        dist = dist_fn(q, node.item.emb)
        if len(heap) < k or dist < heap[0][0]:
            heapq.heappush(heap, (-dist, node.item.id))  # Max-heap via negation
            if len(heap) > k:
                heapq.heappop(heap)

        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist_fn, heap)

        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)

    def insert(self, v: VectorItem) -> None:
        """Insert a vector item."""
        self.root = self._insert(self.root, v, 0)

    def knn(self, q: List[float], k: int, dist_fn) -> List[Tuple[float, int]]:
        """
        Find k-nearest neighbors.
        Returns list of (distance, id) pairs sorted by distance.
        """
        heap: List[Tuple[float, int]] = []  # Using negative distances for max-heap
        self._knn(self.root, q, k, 0, dist_fn, heap)

        # Convert max-heap (with negated distances) to sorted list
        results = [(-dist, id_) for dist, id_ in heap]
        results.sort(key=lambda x: x[0])
        return results

    def rebuild(self, items: List[VectorItem]) -> None:
        """Rebuild tree from scratch."""
        self._destroy(self.root)
        self.root = None
        for v in items:
            self.insert(v)
