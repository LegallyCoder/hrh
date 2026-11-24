"""Brain-free proxies: properties of a model computed WITHOUT any neural data.

From each layer's Gram matrix over the 200 THINGS-EEG2 test images:
  alpha(l)  spectral exponent: power-law decay of the eigenvalues of the
            double-centred Gram (eigenvalue_k ~ k^-alpha), ranks RANK_LO..RANK_HI.
            Theory (P4, P5): alignment with a processing stage peaks when the
            layer's exponent matches the stage's.
  L(l)      low-level retention: RSA with a fixed Gabor-energy model of the images.
  S(l)      semantic retention: RSA with a kernel built from human category
            labels (THINGSplus, 53 high-level categories; Jaccard similarity).

Every choice below (rank range, filter bank, pooling, category set) is fixed
before any held-out model is analysed.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from . import align

RANK_LO, RANK_HI = 2, 50              # skip the top component; stay above the noise floor
GABOR_SIZE = 128                      # images are resized to GABOR_SIZE^2 grayscale
GABOR_WAVELENGTHS = (4.0, 8.0, 16.0, 32.0)
GABOR_ORIENTATIONS = 8
GABOR_GRID = 4                        # energy pooled on a 4x4 grid (as grid4 pooling)
CATEGORY53_URL = "https://osf.io/download/pa9ry/"   # THINGSplus category53_wide-format.tsv


# ---------------------------------------------------------------------------
# Spectral exponent
# ---------------------------------------------------------------------------

def centre(gram: np.ndarray) -> np.ndarray:
    """Double-centred Gram = Gram of mean-centred features."""
    g = gram - gram.mean(0, keepdims=True)
    return g - g.mean(1, keepdims=True)


def spectral_alpha(gram: np.ndarray, lo: int = RANK_LO, hi: int = RANK_HI) -> float:
    e = np.sort(np.linalg.eigvalsh(centre(gram)))[::-1][lo - 1:hi]
    r = np.arange(lo, lo + len(e))
    keep = e > max(e[0], 1e-30) * 1e-10
    if keep.sum() < 5:
        return float("nan")
    return float(-np.polyfit(np.log(r[keep]), np.log(e[keep]), 1)[0])


# ---------------------------------------------------------------------------
# Reference kernels (from images and human labels only)
# ---------------------------------------------------------------------------

def _gray(path: str, size: int) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("L").resize((size, size), Image.BICUBIC)
    x = np.asarray(im, dtype=float) / 255.0
    return x - x.mean()


def gabor_bank(size: int = GABOR_SIZE) -> np.ndarray:
    """Complex Gabor filters in the Fourier domain: (n_filters, size, size)."""
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    out = []
    for lam in GABOR_WAVELENGTHS:
        f0, bw = 1.0 / lam, 0.5 / lam                 # one-octave bandwidth
        for k in range(GABOR_ORIENTATIONS):
            th = np.pi * k / GABOR_ORIENTATIONS
            u0, v0 = f0 * np.cos(th), f0 * np.sin(th)
            out.append(np.exp(-((fx - u0) ** 2 + (fy - v0) ** 2) / (2 * bw ** 2)))
    return np.stack(out)


def gabor_features(paths: list[str], size: int = GABOR_SIZE, grid: int = GABOR_GRID) -> np.ndarray:
    bank = gabor_bank(size)
    feats = []
    for p in paths:
        f = np.fft.fft2(_gray(p, size))
        energy = np.abs(np.fft.ifft2(bank * f[None]))          # (n_filters, size, size)
        n, s = energy.shape[0], size // grid
        pooled = energy.reshape(n, grid, s, grid, s).mean((2, 4))
        feats.append(pooled.ravel())
    x = np.log1p(np.stack(feats) / np.mean(feats))             # compress energy
    return x


def pixel_features(paths: list[str], size: int = 16) -> np.ndarray:
    return np.stack([_gray(p, size).ravel() for p in paths])


def features_to_rdm(x: np.ndarray) -> np.ndarray:
    return align.gram_to_corr_rdm(x @ x.T, x.sum(1), x.shape[1])


def load_categories(path: str, concepts: list[str]) -> np.ndarray:
    """Binary membership matrix (n_concepts, 53) from category53_wide-format.tsv."""
    with open(path) as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    table = {r[0]: np.array(r[2:], float) for r in rows[1:]}
    missing = [c for c in concepts if c not in table]
    if missing:
        raise KeyError(f"concepts without category rows: {missing[:5]}")
    return np.stack([table[c] for c in concepts])


def semantic_rdm(membership: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Jaccard distance between category sets. Returns (rdm, valid) where valid
    marks stimuli that have at least one category."""
    m = membership > 0
    inter = (m[:, None] & m[None]).sum(-1)
    union = (m[:, None] | m[None]).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1.0 - np.where(union > 0, inter / union, 0.0)
    return d, m.any(1)


def ensure_categories(path: str) -> str:
    if not os.path.exists(path):
        from .data import fetch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fetch(CATEGORY53_URL, path)
    return path


def test_concepts(metadata_path: str) -> list[str]:
    meta = np.load(metadata_path, allow_pickle=True).item()
    return [c.split("_", 1)[1] for c in meta["test_img_concepts"]]


# ---------------------------------------------------------------------------
# Per-layer proxies
# ---------------------------------------------------------------------------

def _rsa_masked(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None) -> float:
    if valid is None or valid.all():
        return align.rsa(a, b)
    idx = np.flatnonzero(valid)
    return align.rsa(a[np.ix_(idx, idx)], b[np.ix_(idx, idx)])


def layer_proxies(grams: np.ndarray, rdms: np.ndarray, low_rdm: np.ndarray,
                  sem_rdm: np.ndarray, sem_valid: np.ndarray) -> dict:
    """grams: (L, N, N) raw Grams; rdms: (L, N, N) model RDMs of the same layers."""
    return {"alpha": np.array([spectral_alpha(g) for g in grams]),
            "L": np.array([align.rsa(r, low_rdm) for r in rdms]),
            "S": np.array([_rsa_masked(r, sem_rdm, sem_valid) for r in rdms])}


def summarize(p: dict, n_depth: int = 8) -> dict:
    """Model-level proxies used by the law (hrh/law.py)."""
    n = len(p["alpha"])
    depth = np.linspace(0, 1, n) if n > 1 else np.zeros(1)
    grid = np.linspace(0, 1, n_depth)
    prof = {k: np.interp(grid, depth, v).round(4).tolist() for k, v in p.items()}
    a = p["alpha"][np.isfinite(p["alpha"])]
    return {"n_layers": n, "L_max": float(np.nanmax(p["L"])), "S_max": float(np.nanmax(p["S"])),
            "alpha_min": float(a.min()), "alpha_max": float(a.max()),
            "alpha_end": float(p["alpha"][-1]), "alpha": p["alpha"].round(4).tolist(),
            "L": p["L"].round(4).tolist(), "S": p["S"].round(4).tolist(), "profile": prof}
