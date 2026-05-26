"""
=============================================================================
algorithms.py
=============================================================================
Contains:
  • AHP_pairwise_comparison()  – subjective weighting via Analytic Hierarchy Process
  • calculate_gini_index()     – objective weighting via Gini Index
  • fuse_weights()             – blend Ws and Wo with β coefficient
=============================================================================
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Saaty's Random Consistency Index table (index = matrix size n)
# ---------------------------------------------------------------------------
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def AHP_pairwise_comparison(
    pairwise_matrix: np.ndarray,
    cr_threshold: float = 0.10,
    verbose: bool = True
) -> Optional[np.ndarray]:
    """
    Analytic Hierarchy Process – Step 1: Subjective Weighting
    ----------------------------------------------------------
    Given a square pairwise-comparison matrix A (n×n), returns the
    normalised priority / weight vector  Ws = [ws1, ws2, …, wsn].

    Parameters
    ----------
    pairwise_matrix : np.ndarray  (n × n)
        Saaty scale matrix where a[i][j] = importance of criterion i
        relative to criterion j.  a[j][i] = 1 / a[i][j].
    cr_threshold : float
        Maximum acceptable Consistency Ratio (default = 0.10 = 10 %).
    verbose : bool
        Print intermediate steps if True.

    Returns
    -------
    Ws : np.ndarray  (n,)
        Subjective weight vector, sums to 1.
        Returns None if the matrix is inconsistent beyond the threshold.
    """
    A = np.array(pairwise_matrix, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Pairwise matrix must be square.")

    col_sums = A.sum(axis=0)
    A_norm   = A / col_sums

    Ws = A_norm.mean(axis=1)

    weighted = A @ Ws
    lambda_max = np.mean(weighted / Ws)

    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0.0

    if verbose:
        print("── AHP Results ──────────────────────────────────────")
        print(f"   λ_max = {lambda_max:.4f}")
        print(f"   CI    = {CI:.4f}")
        print(f"   RI    = {RI:.4f}")
        print(f"   CR    = {CR:.4f}  ({'✓ consistent' if CR <= cr_threshold else '✗ INCONSISTENT'})")
        print(f"   Ws    = {np.round(Ws, 4)}")
        print("─────────────────────────────────────────────────────")

    if CR > cr_threshold:
        print(f"[AHP WARNING] CR={CR:.4f} exceeds threshold {cr_threshold}. "
              "Revise the pairwise matrix.")
        return None

    return Ws


def calculate_gini_index(
    edge_criteria_matrix: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Objective Weighting via Gini Index – Step 2
    --------------------------------------------
    Analyses the spread (inequality) of each criterion across all edges
    in the graph.
    """
    X = np.array(edge_criteria_matrix, dtype=float)
    M, n = X.shape

    gini_scores = np.zeros(n)

    for j in range(n):
        col = np.sort(X[:, j])
        ranks   = np.arange(1, M + 1)
        col_sum = col.sum()

        if col_sum == 0:
            gini_scores[j] = 0.0
        else:
            gini_scores[j] = (2.0 * (ranks * col).sum()) / (M * col_sum) - (M + 1) / M

    gini_scores = np.clip(gini_scores, 0, None)

    total = gini_scores.sum()
    Wo    = gini_scores / total if total != 0 else np.ones(n) / n

    if verbose:
        print("── Gini Index Results ───────────────────────────────")
        print(f"   Raw Gini scores : {np.round(gini_scores, 4)}")
        print(f"   Wo (normalised) : {np.round(Wo, 4)}")
        print("─────────────────────────────────────────────────────")

    return Wo


def fuse_weights(
    Ws: np.ndarray,
    Wo: np.ndarray,
    beta: float = 0.5,
    verbose: bool = True
) -> np.ndarray:
    """
    Weight Fusion – Step 3
    ----------------------
    W = β · Ws + (1 − β) · Wo
    """
    Ws = np.array(Ws, dtype=float)
    Wo = np.array(Wo, dtype=float)

    if Ws.shape != Wo.shape:
        raise ValueError("Ws and Wo must have the same length.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")

    W = beta * Ws + (1.0 - beta) * Wo
    W /= W.sum()

    if verbose:
        print("── Weight Fusion Results ────────────────────────────")
        print(f"   β  = {beta}")
        print(f"   Ws = {np.round(Ws, 4)}")
        print(f"   Wo = {np.round(Wo, 4)}")
        print(f"   W  = {np.round(W,  4)}")
        print("─────────────────────────────────────────────────────")

    return W


if __name__ == "__main__":
    print("\n========== MODULE SELF-TEST ==========\n")

    A = np.array([
        [1,   3,   5,   7  ],
        [1/3, 1,   3,   5  ],
        [1/5, 1/3, 1,   3  ],
        [1/7, 1/5, 1/3, 1  ],
    ])
    Ws = AHP_pairwise_comparison(A)

    np.random.seed(42)
    edge_data = np.array([
        [10, 5,  0.1, 20],
        [15, 8,  0.4, 35],
        [8,  3,  0.05,15],
        [20, 12, 0.8, 50],
        [12, 6,  0.2, 25],
        [18, 10, 0.6, 45],
    ], dtype=float)

    Wo = calculate_gini_index(edge_data)

    if Ws is not None:
        W = fuse_weights(Ws, Wo, beta=0.6)
        print(f"\nFinal Comprehensive Weight Vector W: {np.round(W, 4)}")