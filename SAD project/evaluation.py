"""
=============================================================================
evaluation.py  —  Evaluation & Experiment Framework
=============================================================================
Contains:
  • generate_test_graphs()   – random graphs with N nodes
  • run_all_comparisons()    – benchmark all 4 algorithms
  • generate_plots()         – matplotlib plots of results
=============================================================================
"""

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Dict, Tuple

from algorithms import (
    AHP_pairwise_comparison,
    calculate_gini_index,
    fuse_weights
)

from optimization import (
    dijkstra_standard,
    dijkstra_ahp_hurwicz,
    dijkstra_gini_hurwicz,
    dijkstra_full_hybrid
)

# ---------------------------------------------------------------------------
# Fixed AHP pairwise matrix (used for all experiments)
# 4 criteria: time, distance, risk, cost
# ---------------------------------------------------------------------------
PAIRWISE_MATRIX = np.array([
    [1,   3,   5,   7  ],
    [1/3, 1,   3,   5  ],
    [1/5, 1/3, 1,   3  ],
    [1/7, 1/5, 1/3, 1  ],
])

ALPHA = 0.5
BETA  = 0.6


# ===========================================================================
# Graph generator
# ===========================================================================

def generate_test_graphs(
    N: int,
    edge_probability: float = 0.15,
    n_criteria: int = 4,
    seed: int = 42
) -> Tuple[Dict, int, int]:

    """
    Generate a random directed graph with N nodes.

    Each edge carries shape-(2, n_criteria) array:
        row 0 = x_min per criterion
        row 1 = x_max per criterion

    Returns
    -------
    graph  : adjacency dict
    source : int
    target : int
    """

    rng = np.random.default_rng(seed)
    G = nx.gnp_random_graph(N, edge_probability, directed=True, seed=seed)

    # Ensure connectivity
    for i in range(N - 1):
        G.add_edge(i, i + 1)

    graph = {}

    for u in G.nodes():
        graph[u] = {}

        for v in G.successors(u):
            x_min = rng.uniform(1, 10, size=n_criteria)
            x_max = x_min + rng.uniform(0, 10, size=n_criteria)

            graph[u][v] = np.stack([x_min, x_max])

    return graph, 0, N - 1


# ===========================================================================
# Benchmark runner
# ===========================================================================

def run_all_comparisons(
    graph: Dict,
    source: int,
    target: int,
    Ws: np.ndarray,
    Wo: np.ndarray,
    W: np.ndarray,
    alpha: float = ALPHA
) -> Dict:

    """
    Run all 4 algorithms on the given graph and collect metrics.
    """

    algorithms = {
        "Standard Dijkstra": lambda: dijkstra_standard(graph, source, target),

        "AHP+Hurwicz+Dijkstra":
            lambda: dijkstra_ahp_hurwicz(graph, source, target, Ws, alpha),

        "Gini+Hurwicz+Dijkstra":
            lambda: dijkstra_gini_hurwicz(graph, source, target, Wo, alpha),

        "Full Hybrid":
            lambda: dijkstra_full_hybrid(graph, source, target, W, alpha),
    }

    results = {}

    for name, fn in algorithms.items():
        t0 = time.perf_counter()

        cost, path = fn()

        elapsed = time.perf_counter() - t0

        results[name] = {
            "cost": cost,
            "path_length": len(path),
            "time_ms": elapsed * 1000,
            "path": path,
        }

    # Optimality gap
    costs = [v["cost"] for v in results.values() if v["cost"] != float('inf')]
    best = min(costs) if costs else 1.0

    for v in results.values():
        v["optimality_gap"] = (
            (v["cost"] - best) / best if best > 0 else 0.0
        )

    return results


# ===========================================================================
# Experiment across graph sizes
# ===========================================================================

def run_experiments(
    sizes: List[int] = [10, 50, 100, 500, 1000],
    n_criteria: int = 4,
    verbose: bool = True
) -> Dict:

    """
    Run all algorithms across multiple graph sizes.
    """

    Ws = AHP_pairwise_comparison(PAIRWISE_MATRIX, verbose=False)

    assert Ws is not None, "AHP matrix is inconsistent."

    rng = np.random.default_rng(0)
    dummy_edges = rng.uniform(1, 20, size=(50, n_criteria))

    Wo = calculate_gini_index(dummy_edges, verbose=False)

    W = fuse_weights(Ws, Wo, beta=BETA, verbose=False)

    all_results = {}

    for N in sizes:

        if verbose:
            print(f"\n── N = {N} nodes ─────────────────────────────────────")

        graph, source, target = generate_test_graphs(N, seed=42)

        edge_vals = []

        for u in graph:
            for v in graph[u]:
                edge_vals.append(graph[u][v].mean(axis=0))

        Wo_local = calculate_gini_index(
            np.array(edge_vals),
            verbose=False
        )

        W_local = fuse_weights(
            Ws,
            Wo_local,
            beta=BETA,
            verbose=False
        )

        res = run_all_comparisons(
            graph,
            source,
            target,
            Ws,
            Wo_local,
            W_local
        )

        if verbose:
            for algo, m in res.items():

                print(
                    f"   {algo:30s}  "
                    f"cost={m['cost']:9.3f}  "
                    f"path_len={m['path_length']:4d}  "
                    f"time={m['time_ms']:.3f}ms  "
                    f"gap={m['optimality_gap']*100:.1f}%"
                )

        all_results[N] = res

    return all_results


# ===========================================================================
# Plotting
# ===========================================================================

ALGO_COLORS = {
    "Standard Dijkstra": "#e74c3c",
    "AHP+Hurwicz+Dijkstra": "#3498db",
    "Gini+Hurwicz+Dijkstra": "#72e65b",
    "Full Hybrid": "#e4a929",
}

ALGO_MARKERS = {
    "Standard Dijkstra": "o",
    "AHP+Hurwicz+Dijkstra": "s",
    "Gini+Hurwicz+Dijkstra": "^",
    "Full Hybrid": "D",
}


def generate_plots(
    all_results: Dict,
    save_path: str = "comparison_plots.png"
) -> None:

    """
    Generate and save comparison plots.
    """

    sizes = sorted(all_results.keys())

    algos = list(next(iter(all_results.values())).keys())

    metrics = {
        "time_ms": ("Execution Time (ms)", "log"),
        "cost": ("Path Cost (C_ij)", "linear"),
        "path_length": ("Path Length (hops)", "linear"),
        "optimality_gap": ("Optimality Gap (%)", "linear"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(
        "Multi-Criteria Route Optimisation\nAlgorithm Comparison",
        fontsize=15,
        fontweight='bold',
        y=1.01
    )

    for ax, (metric, (ylabel, yscale)) in zip(axes.flat, metrics.items()):

        for algo in algos:

            vals = []

            for N in sizes:
                v = all_results[N][algo][metric]

                vals.append(v * 100 if metric == "optimality_gap" else v)

            ax.plot(
                sizes,
                vals,
                label=algo,
                color=ALGO_COLORS[algo],
                marker=ALGO_MARKERS[algo],
                linewidth=2,
                markersize=6
            )

        ax.set_xlabel("Graph Size N (nodes)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        ax.set_title(
            ylabel,
            fontsize=12,
            fontweight='bold'
        )

        ax.set_yscale(yscale)
        ax.set_xscale("log")

        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        ax.set_xticks(sizes)

        ax.grid(True, alpha=0.3)

        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches='tight'
    )

    print(f"\n[Plots saved → {save_path}]")

    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    SIZES = [10, 50, 100, 500, 1000]

    print("=" * 60)
    print("  Experiment Runner")
    print("=" * 60)

    all_results = run_experiments(
        sizes=SIZES,
        verbose=True
    )

    generate_plots(
        all_results,
        save_path="results/comparison_plots.png"
    )

    # Save CSV
    import os
    import csv

    os.makedirs("results", exist_ok=True)

    with open("results/results_table.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "N",
            "Algorithm",
            "Cost",
            "PathLength",
            "TimeMs",
            "OptimalityGap%"
        ])

        for N in sorted(all_results):

            for algo, m in all_results[N].items():

                writer.writerow([
                    N,
                    algo,
                    round(m["cost"], 4),
                    m["path_length"],
                    round(m["time_ms"], 4),
                    round(m["optimality_gap"] * 100, 2)
                ])

    print("[Results saved → results/results_table.csv]")