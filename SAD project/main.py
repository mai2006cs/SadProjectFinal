"""
=============================================================================
main.py  —  Subject N°1 Entry Point
=============================================================================
Run this file to execute the full 5-step pipeline on a demo graph and then
launch the full experimental evaluation.

Usage:
    python main.py
=============================================================================
"""

import os
import numpy as np

# ── Algorithms module ───────────────────────────────────────────────────────
from algorithms import (
    AHP_pairwise_comparison,
    calculate_gini_index,
    fuse_weights,
)

# ── Optimization modules ────────────────────────────────────────────────────
from optimization import (
    dijkstra_standard,
    dijkstra_ahp_hurwicz,
    dijkstra_gini_hurwicz,
    dijkstra_full_hybrid,
)
from evaluation import run_experiments, generate_plots


# ===========================================================================
# Configuration
# ===========================================================================
N_CRITERIA = 4              # time, distance, risk, cost
ALPHA      = 0.5            # balanced optimism (Hurwicz)
BETA       = 0.6            # weight fusion: 60 % AHP, 40 % Gini

PAIRWISE_MATRIX = np.array([
    [1,   3,   5,   7  ],
    [1/3, 1,   3,   5  ],
    [1/5, 1/3, 1,   3  ],
    [1/7, 1/5, 1/3, 1  ],
])

# Small demo graph (10 nodes) for step-by-step demonstration
DEMO_GRAPH = {
    0: {1: np.array([[5, 10, 0.1, 20], [10, 15, 0.3, 30]]),
        2: np.array([[8, 12, 0.2, 25], [14, 18, 0.5, 40]])},
    1: {3: np.array([[4,  6, 0.1, 15], [8,  10, 0.4, 22]]),
        4: np.array([[6,  9, 0.3, 18], [11, 13, 0.6, 28]])},
    2: {3: np.array([[3,  5, 0.05,10], [6,   8, 0.3, 18]]),
        5: np.array([[7,  9, 0.2, 20], [12,  14, 0.6, 35]])},
    3: {6: np.array([[2,  4, 0.05, 8], [5,   7, 0.2, 15]])},
    4: {6: np.array([[3,  5, 0.1, 10], [7,  9, 0.3, 18]]),
        7: np.array([[9, 11, 0.4, 25], [14, 16, 0.7, 40]])},
    5: {7: np.array([[4,  6, 0.15,12], [8,  10, 0.4, 22]]),
        8: np.array([[5,  7, 0.2, 15], [9,  12, 0.5, 28]])},
    6: {9: np.array([[3,  5, 0.1, 10], [6,   8, 0.3, 18]])},
    7: {9: np.array([[4,  6, 0.15,12], [7,  10, 0.4, 22]])},
    8: {9: np.array([[2,  4, 0.05, 8], [5,   7, 0.2, 15]])},
    9: {},
}
SOURCE, TARGET = 0, 9


def banner(text: str) -> None:
    w = 60
    print("\n" + "═" * w)
    print(f"  {text}")
    print("═" * w)


def main():
    os.makedirs("results", exist_ok=True)

    # ── Step 1: AHP – Subjective Weights ────────────────────────────────────
    banner("STEP 1 — AHP")
    Ws = AHP_pairwise_comparison(PAIRWISE_MATRIX)
    assert Ws is not None, "AHP consistency check failed. Revise the matrix."

    # ── Step 2: Gini Index – Objective Weights ──────────────────────────────
    banner("STEP 2 — Gini Index")
    edge_vals = []
    for u in DEMO_GRAPH:
        for v in DEMO_GRAPH[u]:
            edge_vals.append(DEMO_GRAPH[u][v].mean(axis=0))
    Wo = calculate_gini_index(np.array(edge_vals))

    # ── Step 3: Weight Fusion ───────────────────────────────────────────────
    banner("STEP 3 — Weight Fusion")
    W = fuse_weights(Ws, Wo, beta=BETA)

    # ── Steps 4+5: All 4 Algorithms ────────────────────────────────────────
    banner("STEPS 4+5 — All 4 Algorithms")

    results = {
        "Standard Dijkstra"     : dijkstra_standard(DEMO_GRAPH, SOURCE, TARGET),
        "AHP+Hurwicz+Dijkstra"  : dijkstra_ahp_hurwicz(DEMO_GRAPH, SOURCE, TARGET, Ws, ALPHA),
        "Gini+Hurwicz+Dijkstra" : dijkstra_gini_hurwicz(DEMO_GRAPH, SOURCE, TARGET, Wo, ALPHA),
        "Full Hybrid"           : dijkstra_full_hybrid(DEMO_GRAPH, SOURCE, TARGET, W,  ALPHA),
    }

    print(f"\n{'Algorithm':32s}  {'Cost':>10s}  {'Path'}")
    print("─" * 70)
    for algo, (cost, path) in results.items():
        print(f"  {algo:30s}  {cost:10.4f}  {path}")

    # ── Full Experiments ────────────────────────────────────────────────────
    banner("FULL EXPERIMENTS")
    print("Running on graph sizes N = 10, 50, 100, 500, 1000 …")
    all_results = run_experiments(sizes=[10, 50, 100, 500, 1000], verbose=True)
    generate_plots(all_results, save_path="results/comparison_plots.png")

    print("\n✓ All done!  Check the results/ folder for plots and CSV.")


if __name__ == "__main__":
    main()