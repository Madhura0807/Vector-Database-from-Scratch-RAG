import heapq
import math
import random
from typing import List, Tuple, Dict, Set, Callable
from brute_force import VectorItem


class HNSW:
    """Hierarchical Navigable Small World graph for approximate nearest neighbor search."""

    class Node:
        """HNSW node with multilayer connections."""
        def __init__(self, item: VectorItem, max_lyr: int):
            self.item = item
            self.max_lyr = max_lyr
            self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]

    def __init__(self, m: int = 16, ef_build: int = 200):
        self.G: Dict[int, HNSW.Node] = {}  # Graph: id -> Node
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_point = -1
        self.rng = random.Random(42)  # Fixed seed for reproducibility

    def _rand_level(self) -> int:
        """Generate random level using exponential distribution."""
        u = self.rng.random()
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, layer: int,
                      dist_fn: Callable) -> List[Tuple[float, int]]:
        """
        Greedy search within a single layer.
        Returns sorted list of (distance, id) pairs.
        """
        visited: Set[int] = set()
        # Min-heap for candidates (distance, id)
        candidates: List[Tuple[float, int]] = []
        # Max-heap for found results (using negative distance)
        found: List[Tuple[float, int]] = []

        d0 = dist_fn(q, self.G[ep].item.emb)
        visited.add(ep)
        heapq.heappush(candidates, (d0, ep))
        heapq.heappush(found, (-d0, ep))

        while candidates:
            cd, cid = heapq.heappop(candidates)
            if len(found) >= ef and cd > -found[0][0]:
                break
            if layer >= len(self.G[cid].nbrs):
                continue
            for nid in self.G[cid].nbrs[layer]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist_fn(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(candidates, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        # Convert max-heap (with negated distances) to sorted list
        results = [(-dist, id_) for dist, id_ in found]
        results.sort(key=lambda x: x[0])
        return results

    def _select_neighbors(self, candidates: List[Tuple[float, int]], max_m: int) -> List[int]:
        """Select top M neighbors from candidates."""
        return [cand[1] for cand in candidates[:min(len(candidates), max_m)]]

    def insert(self, item: VectorItem, dist_fn: Callable) -> None:
        """Insert a vector item into the HNSW graph."""
        id_ = item.id
        lvl = self._rand_level()
        self.G[id_] = HNSW.Node(item, lvl)

        if self.entry_point == -1:
            self.entry_point = id_
            self.top_layer = lvl
            return

        ep = self.entry_point
        # Descend from top layer to insertion level
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        # Insert at each layer from lvl down to 0
        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_neighbors(W, max_m)
            self.G[id_].nbrs[lc] = sel

            # Bidirectional connections
            for nid in sel:
                if nid not in self.G:
                    continue
                if len(self.G[nid].nbrs) <= lc:
                    self.G[nid].nbrs.extend([[] for _ in range(lc + 1 - len(self.G[nid].nbrs))])
                conn = self.G[nid].nbrs[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    # Prune by distance
                    ds = []
                    for c in conn:
                        if c in self.G:
                            ds.append((dist_fn(self.G[nid].item.emb, self.G[c].item.emb), c))
                    ds.sort(key=lambda x: x[0])
                    conn.clear()
                    for i in range(min(max_m, len(ds))):
                        conn.append(ds[i][1])

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_point = id_

    def knn(self, q: List[float], k: int, ef: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        """
        Find k-nearest neighbors.
        Returns list of (distance, id) pairs sorted by distance.
        """
        if self.entry_point == -1:
            return []

        ep = self.entry_point
        # Descend from top layer to layer 1
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        # Search at layer 0
        W = self._search_layer(q, ep, max(ef, k), 0, dist_fn)
        if len(W) > k:
            W = W[:k]
        return W

    def remove(self, id_: int) -> None:
        """Remove a node from the graph."""
        if id_ not in self.G:
            return

        # Remove from all neighbor lists
        for nid, node in self.G.items():
            for layer in node.nbrs:
                if id_ in layer:
                    layer.remove(id_)

        # Update entry point if needed
        if self.entry_point == id_:
            self.entry_point = -1
            for nid in self.G:
                if nid != id_:
                    self.entry_point = nid
                    break

        del self.G[id_]

    def get_info(self) -> dict:
        """Get graph structure information."""
        gi = {
            'topLayer': self.top_layer,
            'nodeCount': len(self.G),
            'nodesPerLayer': [],
            'edgesPerLayer': [],
            'nodes': [],
            'edges': []
        }

        max_l = max(self.top_layer + 1, 1)
        gi['nodesPerLayer'] = [0] * max_l
        gi['edgesPerLayer'] = [0] * max_l

        for id_, node in self.G.items():
            gi['nodes'].append({
                'id': id_,
                'metadata': node.item.metadata,
                'category': node.item.category,
                'maxLyr': node.max_lyr
            })

            for lc in range(min(node.max_lyr + 1, max_l)):
                gi['nodesPerLayer'][lc] += 1
                if lc < len(node.nbrs):
                    for nid in node.nbrs[lc]:
                        if id_ < nid:
                            gi['edgesPerLayer'][lc] += 1
                            gi['edges'].append({'src': id_, 'dst': nid, 'lyr': lc})

        return gi

    def size(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.G)
