"""
=============================================================================
optimization.py
=============================================================================
Contains:
  • hurwicz_criterion()         – α-Hurwicz risk resolution (Step 4)
  • dijkstra_standard()         – baseline single-criterion Dijkstra
  • dijkstra_ahp_hurwicz()      – Steps 1, 4, 5  (AHP + Hurwicz + Dijkstra)
  • dijkstra_gini_hurwicz()     – Steps 2, 4, 5  (Gini + Hurwicz + Dijkstra)
  • dijkstra_full_hybrid()      – Steps 1-5 (AHP + Gini + Hurwicz + Dijkstra)
=============================================================================
"""

import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Graph representation
# ---------------------------------------------------------------------------
# Graph is stored as an adjacency dict:
#   graph[u][v] = np.ndarray (n,)  →  n criteria values for edge (u, v)
#               OR
#   graph[u][v] = float            →  single scalar weight (baseline mode)
# ---------------------------------------------------------------------------


def hurwicz_criterion(
    x_min: np.ndarray,
    x_max: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    α-Hurwicz Risk Resolution – Step 4
    ------------------------------------
    For each criterion k on an edge, compute the expected performance value V:
        V_k = α · x_min_k  +  (1 − α) · x_max_k

    Parameters
    ----------
    x_min  : np.ndarray (n,)  – best-case  (minimum cost) per criterion
    x_max  : np.ndarray (n,)  – worst-case (maximum cost) per criterion
    alpha  : float            – optimism coefficient ∈ [0,1]
                                α=0 → purely pessimistic (Maximin / minimax cost)
                                α=1 → purely optimistic  (best case only)

    Returns
    -------
    V : np.ndarray (n,)  – Hurwicz value per criterion
    """
    return alpha * np.array(x_min) + (1.0 - alpha) * np.array(x_max)


def _compute_edge_cost(
    criteria_values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    use_hurwicz: bool = True
) -> float:
    """
    Internal helper: compute scalar edge cost C_ij from criteria values + weights.

    If use_hurwicz=True, criteria_values must be shape (2, n): [x_min, x_max].
    If use_hurwicz=False, criteria_values is already the resolved shape (n,).

    C_ij = Σ_k  w_k · V_ij^k
    """
    if use_hurwicz:
        x_min = criteria_values[0]
        x_max = criteria_values[1]
        V = hurwicz_criterion(x_min, x_max, alpha)
    else:
        V = np.array(criteria_values, dtype=float)

    return float(np.dot(weights, V))


def _dijkstra(
    graph: Dict,
    source: int,
    target: int,
    cost_fn
) -> Tuple[float, List[int]]:
    """
    Generic Dijkstra.  cost_fn(u, v) returns the scalar edge cost.

    Returns
    -------
    (total_cost, path)
    """
    dist    = {source: 0.0}
    prev    = {source: None}
    visited = set()
    heap    = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v in graph.get(u, {}):
            if v in visited:
                continue
            nd = d + cost_fn(u, v)
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if target not in dist:
        return float('inf'), []

    path, node = [], target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return dist[target], path


def dijkstra_standard(
    graph: Dict,
    source: int,
    target: int,
    criterion_index: int = 0
) -> Tuple[float, List[int]]:
    """
    Baseline Dijkstra using only one criterion.
    """
    def cost_fn(u, v):
        val = graph[u][v]
        if np.isscalar(val):
            return float(val)
        val = np.array(val)
        if val.ndim == 1:
            return float(val[criterion_index])
        return float(val[:, criterion_index].mean())

    return _dijkstra(graph, source, target, cost_fn)


def dijkstra_ahp_hurwicz(
    graph: Dict,
    source: int,
    target: int,
    Ws: np.ndarray,
    alpha: float = 0.5
) -> Tuple[float, List[int]]:
    Ws = np.array(Ws, dtype=float)

    def cost_fn(u, v):
        edge = np.array(graph[u][v])
        return _compute_edge_cost(edge, Ws, alpha, use_hurwicz=True)

    return _dijkstra(graph, source, target, cost_fn)


def dijkstra_gini_hurwicz(
    graph: Dict,
    source: int,
    target: int,
    Wo: np.ndarray,
    alpha: float = 0.5
) -> Tuple[float, List[int]]:
    Wo = np.array(Wo, dtype=float)

    def cost_fn(u, v):
        edge = np.array(graph[u][v])
        return _compute_edge_cost(edge, Wo, alpha, use_hurwicz=True)

    return _dijkstra(graph, source, target, cost_fn)


def dijkstra_full_hybrid(
    graph: Dict,
    source: int,
    target: int,
    W: np.ndarray,
    alpha: float = 0.5
) -> Tuple[float, List[int]]:
    W = np.array(W, dtype=float)

    def cost_fn(u, v):
        edge = np.array(graph[u][v])
        return _compute_edge_cost(edge, W, alpha, use_hurwicz=True)

    return _dijkstra(graph, source, target, cost_fn)


if __name__ == "__main__":
    import numpy as np

    print("\n========== MODULE SELF-TEST ==========\n")

    graph = {
        0: {
            1: np.array([[5, 10, 0.1, 20], [10, 15, 0.3, 30]]),
            2: np.array([[8, 12, 0.2, 25], [14, 18, 0.5, 40]]),
        },
        1: {
            3: np.array([[4, 6,  0.1, 15], [8,  10, 0.4, 22]]),
        },
        2: {
            3: np.array([[3, 5,  0.05,10], [6,  8,  0.3, 18]]),
            4: np.array([[7, 9,  0.2, 20], [12, 14, 0.6, 35]]),
        },
        3: {
            4: np.array([[2, 4,  0.05, 8], [5,  7,  0.2, 15]]),
        },
        4: {}
    }

    W  = np.array([0.35, 0.25, 0.20, 0.20])
    Ws = np.array([0.40, 0.28, 0.18, 0.14])
    Wo = np.array([0.28, 0.21, 0.24, 0.27])
    alpha = 0.5

    cost1, path1 = dijkstra_standard(graph, 0, 4)
    cost2, path2 = dijkstra_ahp_hurwicz(graph, 0, 4, Ws, alpha)
    cost3, path3 = dijkstra_gini_hurwicz(graph, 0, 4, Wo, alpha)
    cost4, path4 = dijkstra_full_hybrid(graph, 0, 4, W,  alpha)

    print(f"  Standard Dijkstra         cost={cost1:.4f}  path={path1}")
    print(f"  AHP+Hurwicz+Dijkstra      cost={cost2:.4f}  path={path2}")
    print(f"  Gini+Hurwicz+Dijkstra     cost={cost3:.4f}  path={path3}")
    print(f"  Full Hybrid               cost={cost4:.4f}  path={path4}")