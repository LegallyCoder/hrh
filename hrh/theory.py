"""Numerical companion to the theory section (deep linear networks, population loss).

Setting: whitened input x in R^d (Cov x = I); target y = M* x with task subspace
S = rowspace(M*), rank r < d; network h_t = W_t ... W_1 x;
objective  ||M* - W_T...W_1||_F^2 + lam * sum_t ||W_t||_F^2   (weight decay).

For a linear map h = A x, the input-space Gram G = A^T A determines population
linear CKA:  CKA(A, B) = <G_A, G_B>_F / (||G_A||_F ||G_B||_F).

Results checked here (proofs in the paper, Appendix A):
  P1  nuisance erasure   at any stationary point with lam > 0: W_1 = W_1 P_S
  P2  power collapse     at any stationary point: G_t = (G_T)^{t/T}
  T1  trajectory ceiling CKA(h_t, b) <= rho_S(b) = ||P_S G_b P_S|| / ||G_b|| for all t
  P3  multi-readout      adding a readout at layer 1 with task subspace S1
                         raises the ceiling to rho_{S + S1}; reconstruction (S1 = R^d) removes it

Anisotropic inputs (Cov x = Sigma). Population CKA becomes the cosine of
WHITENED Grams  G~ = Sigma^{1/2} A^T A Sigma^{1/2}, whose eigenvalues are the
variances of the representation.
  P4  spectral matching  if G~_A ~ Sigma^a and G~_B ~ Sigma^b, CKA depends on a - b
                         only through the spectrum of Sigma, is unimodal in a and
                         peaks at a = b (an inverted U in "how much input spectrum is kept")
  P5  spectral trajectory at a stationary point with target M = Sigma^c (a spectral
                         function of the input statistics), layer t has
                         G~_t ~ Sigma^{1 + 2c t/T}: the exponent runs linearly from the
                         input's (1) to the objective's endpoint (1 + 2c). Pixel
                         reconstruction (c = 0) never leaves the input spectrum; a
                         whitened latent target (c = -1/2) sweeps the whole range [0, 1].
                         Width does not enter, and depth only refines the grid.
"""
from __future__ import annotations

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def proj(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis)
    return q @ q.T


def rho(p: np.ndarray, g: np.ndarray) -> float:
    return float(np.linalg.norm(p @ g @ p) / np.linalg.norm(g))


def psd_power(g: np.ndarray, a: float) -> np.ndarray:
    w, v = np.linalg.eigh((g + g.T) / 2)
    w = np.clip(w, 0, None)
    return (v * w ** a) @ v.T


def grams(ws: list[np.ndarray]) -> list[np.ndarray]:
    out, p = [], np.eye(ws[0].shape[1])
    for w in ws:
        p = w @ p
        out.append(p.T @ p)
    return out


def make_problem(d=24, r=4, m=6, seed=0):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    s_basis = basis[:, :r]
    m_star = rng.normal(size=(m, r)) @ s_basis.T                 # rowspace = S
    # "early brain": full-rank, generic (retinotopic-like) -> energy mostly off S
    b_early = rng.normal(size=(d, d)) / np.sqrt(d)
    return dict(d=d, r=r, m_star=m_star, P=s_basis @ s_basis.T,
                G_early=b_early.T @ b_early, G_mid=m_star.T @ m_star, rng=rng)


def train(prob, depth=4, width=32, lam=1e-2, lr=2e-2, steps=20000, recon=0.0,
          n_log=120, seed=1):
    """Gradient descent on the population objective; returns a log dict.
    recon > 0 adds  recon * ||I - R W_1||^2 + lam ||R||^2  (a readout at layer 1)."""
    rng = np.random.default_rng(seed)
    d, m_star = prob["d"], prob["m_star"]
    dims = [d] + [width] * (depth - 1) + [m_star.shape[0]]
    ws = [rng.normal(size=(dims[i + 1], dims[i])) / np.sqrt(dims[i]) for i in range(depth)]
    ws_init = [w.copy() for w in ws]
    R = rng.normal(size=(d, width)) / np.sqrt(width)
    log = {k: [] for k in ("step", "loss", "cka_early_l1", "cka_early_best", "cka_mid_top",
                           "nuisance_l1", "collapse_err")}
    log_at = set(np.unique(np.geomspace(1, steps, n_log).astype(int)).tolist()) | {0}
    for it in range(steps + 1):
        prods = [np.eye(d)]
        for w in ws:
            prods.append(w @ prods[-1])
        err = m_star - prods[-1]                                 # (m, d)
        if it in log_at:
            gs = [p.T @ p for p in prods[1:]]
            gT = gs[-1]
            log["step"].append(it)
            log["loss"].append(float(np.sum(err ** 2)))
            log["cka_early_l1"].append(cos(gs[0], prob["G_early"]))
            log["cka_early_best"].append(max(cos(g, prob["G_early"]) for g in gs))
            log["cka_mid_top"].append(cos(gT, prob["G_mid"]))
            P = prob["P"]
            log["nuisance_l1"].append(float(np.linalg.norm(gs[0] - P @ gs[0] @ P)
                                            / np.linalg.norm(gs[0])))
            log["collapse_err"].append(max(
                float(np.linalg.norm(gs[t] - psd_power(gT, (t + 1) / depth))
                      / np.linalg.norm(gs[t])) for t in range(depth)))
        # backward: dL/dW_t = -2 A_t^T err B_t^T + 2 lam W_t
        grads = []
        upstream = np.eye(m_star.shape[0])
        for t in reversed(range(depth)):
            grads.append(-2 * upstream.T @ err @ prods[t].T + 2 * lam * ws[t])
            upstream = upstream @ ws[t]
        grads = grads[::-1]
        if recon > 0:
            rerr = np.eye(d) - R @ ws[0]
            grads[0] += -2 * recon * R.T @ rerr
            R -= lr * (-2 * recon * rerr @ ws[0].T + 2 * lam * R)
        for w, g in zip(ws, grads):
            w -= lr * g
    log = {k: np.array(v) for k, v in log.items()}
    log["ws"] = ws
    log["ws_init"] = ws_init
    log["ceiling_early"] = rho(prob["P"], prob["G_early"])
    return log


# ---------------------------------------------------------------------------
# Anisotropic inputs: spectral matching (P4) and spectral trajectories (P5)
# ---------------------------------------------------------------------------

def spectral_cka(eigs: np.ndarray, a: float, b: float) -> float:
    """CKA between whitened Grams Sigma^a and Sigma^b, with Sigma = diag(eigs)."""
    x, y = eigs ** a, eigs ** b
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def make_spectrum(d: int = 16, decay: float = 1.0, seed: int = 0) -> np.ndarray:
    """Sigma with eigenvalues k^-decay (natural images: roughly 1/rank) in a random basis."""
    rng = np.random.default_rng(seed)
    u = np.linalg.qr(rng.normal(size=(d, d)))[0]
    return (u * np.arange(1, d + 1, dtype=float) ** -decay) @ u.T


def train_aniso(sigma: np.ndarray, target: np.ndarray, depth: int = 4, width: int = 32,
                lam: float = 1e-3, lr: float = 0.02, steps: int = 40000,
                seed: int = 1) -> list[np.ndarray]:
    """Gradient descent on E||Mx - W_T..W_1 x||^2 + lam sum ||W_t||^2 with Cov x = Sigma,
    i.e. tr((M - P) Sigma (M - P)^T) + weight decay. Orthogonal initialization."""
    rng = np.random.default_rng(seed)
    d, m = sigma.shape[0], target.shape[0]
    dims = [d] + [width] * (depth - 1) + [m]
    ws = []
    for i in range(depth):
        k = max(dims[i + 1], dims[i])
        ws.append(np.linalg.qr(rng.normal(size=(k, k)))[0][:dims[i + 1], :dims[i]])
    for _ in range(steps):
        prods = [np.eye(d)]
        for w in ws:
            prods.append(w @ prods[-1])
        err = (target - prods[-1]) @ sigma
        grads, up = [], np.eye(m)
        for t in reversed(range(depth)):
            grads.append(-2 * up.T @ err @ prods[t].T + 2 * lam * ws[t])
            up = up @ ws[t]
        for w, g in zip(ws, grads[::-1]):
            w -= lr * g
    return ws


def whitened_grams(ws: list[np.ndarray], sigma: np.ndarray) -> list[np.ndarray]:
    s = psd_power(sigma, 0.5)
    return [s @ g @ s for g in grams(ws)]


def spectral_exponent(g: np.ndarray, lo: int = 1, hi: int | None = None) -> float:
    """Power-law exponent of the eigenvalue decay of a PSD matrix over ranks lo..hi
    (eigenvalue_k ~ k^-alpha; returns alpha)."""
    e = np.sort(np.linalg.eigvalsh((g + g.T) / 2))[::-1]
    hi = hi or len(e)
    e = e[lo - 1:hi]
    r = np.arange(lo, lo + len(e))
    keep = e > e[0] * 1e-8
    return float(-np.polyfit(np.log(r[keep]), np.log(e[keep]), 1)[0])
