"""Alignment metrics between two representations of the same N stimuli.

Everything operates on N x N objects (Gram / kernel / distance matrices), never
on raw features, so brains and nets of any dimensionality are comparable and
features never need to be stored (see features.py: we persist Grams only).

Three metric families are reported side by side, so no result can be blamed on
one metric:
  * RSA  — Spearman between RDM upper triangles (neuroscience standard)
  * CKA  — debiased linear CKA (Song et al. 2012 HSIC_1 estimator); the
           debiased form matters: biased CKA inflates with dimensionality / N
  * mkNN — mutual k-nearest-neighbour overlap (Huh et al. 2024, the PRH metric)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Conversions: Gram (raw features) -> derived matrices
# ---------------------------------------------------------------------------

def gram_to_corr_rdm(gram: np.ndarray, sums: np.ndarray, dim: int) -> np.ndarray:
    """Correlation-distance RDM from a raw-feature Gram matrix.

    Row-centering each feature vector x_i -> x_i - mean(x_i) gives
    <x_i - m_i, x_j - m_j> = G_ij - s_i s_j / d   (s_i = sum of x_i).
    """
    gc = gram - np.outer(sums, sums) / dim
    nrm = np.sqrt(np.clip(np.diag(gc), 1e-12, None))
    return 1.0 - np.clip(gc / np.outer(nrm, nrm), -1.0, 1.0)


def gram_to_cos_dist(gram: np.ndarray) -> np.ndarray:
    nrm = np.sqrt(np.clip(np.diag(gram), 1e-12, None))
    return 1.0 - np.clip(gram / np.outer(nrm, nrm), -1.0, 1.0)


def center(k: np.ndarray) -> np.ndarray:
    n = len(k)
    h = np.eye(n) - 1.0 / n
    return h @ k @ h


def dist_to_kernel(d: np.ndarray) -> np.ndarray:
    """Classical-MDS kernel from a (squared-Euclidean-like) distance matrix."""
    return -0.5 * center(d)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(len(m), 1)]


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks, vectorized (same algorithm as scipy.stats.rankdata)."""
    a = np.asarray(a, float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), np.intp)
    inv[sorter] = np.arange(len(a))
    sa = a[sorter]
    obs = np.r_[True, sa[1:] != sa[:-1]]
    dense = np.cumsum(obs)[inv]
    count = np.r_[np.flatnonzero(obs), len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def rsa(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    return pearson(rankdata(upper(rdm_a)), rankdata(upper(rdm_b)))


def hsic_unbiased(k: np.ndarray, q: np.ndarray) -> float:
    """Song et al. (2012) unbiased HSIC estimator."""
    n = len(k)
    k = k.copy()
    q = q.copy()
    np.fill_diagonal(k, 0.0)
    np.fill_diagonal(q, 0.0)
    ks, ls = k.sum(0), q.sum(0)
    t1 = np.sum(k * q)
    t2 = ks.sum() * ls.sum() / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * (ks @ ls)
    return float((t1 + t2 - t3) / (n * (n - 3)))


def cka_debiased(k: np.ndarray, q: np.ndarray) -> float:
    kl = hsic_unbiased(k, q)
    kk = hsic_unbiased(k, k)
    ll = hsic_unbiased(q, q)
    den = np.sqrt(max(kk, 0.0) * max(ll, 0.0))
    return kl / den if den > 0 else float("nan")


def knn_sets(d: np.ndarray, k: int) -> np.ndarray:
    d = d.copy()
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def mutual_knn(d_a: np.ndarray, d_b: np.ndarray, k: int = 10) -> float:
    """Huh et al. (2024): mean fraction of shared k-NN per stimulus."""
    na, nb = knn_sets(d_a, k), knn_sets(d_b, k)
    n = len(d_a)
    ma = np.zeros((n, n), bool)
    mb = np.zeros((n, n), bool)
    rows = np.arange(n)[:, None]
    ma[rows, na] = True
    mb[rows, nb] = True
    return float((ma & mb).sum(1).mean() / k)


def all_metrics(rdm_a: np.ndarray, rdm_b: np.ndarray, k: int = 10) -> dict:
    return {
        "rsa": rsa(rdm_a, rdm_b),
        "cka": cka_debiased(dist_to_kernel(rdm_a), dist_to_kernel(rdm_b)),
        "mknn": mutual_knn(rdm_a, rdm_b, k),
    }
