"""Brain side: THINGS-EEG2 -> per-time-bin RDMs and noise ceilings.

* Time is kept relative to stimulus onset (epochs run -200..800 ms).
* Repetitions are kept rather than averaged away, which gives
    - cross-validated distances (unbiased: E[d_ij] does not grow with noise,
      so noisy late bins are not mistaken for dissimilar ones),
    - a split-half noise ceiling per time bin,
    - two independent rep-halves for unbiased best-layer selection.
* The pre-stimulus baseline is retained as a built-in null: any metric that
  reports alignment at t < 0 is broken.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np

from . import align


@dataclasses.dataclass
class SubjectRDMs:
    subject: str
    times_ms: np.ndarray          # (T,) relative to stimulus onset
    full: np.ndarray              # (T, N, N) cv-RDMs from all reps
    half_a: np.ndarray            # (T, N, N) cv-RDMs from rep-half A (layer selection)
    half_b: np.ndarray            # (T, N, N) cv-RDMs from rep-half B (evaluation)
    split_half_r: np.ndarray      # (T,) Spearman-Brown corrected reliability
    ceiling: np.ndarray           # (T,) sqrt(reliability): max attainable RSA

    def save(self, path: str) -> None:
        d = dataclasses.asdict(self)
        for k in ("full", "half_a", "half_b"):
            d[k] = d[k].astype(np.float32)             # halves disk; precision is ample
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str) -> "SubjectRDMs":
        z = np.load(path, allow_pickle=False)
        return cls(**{f.name: (str(z[f.name]) if f.name == "subject"
                               else z[f.name].astype(np.float64))
                      for f in dataclasses.fields(cls)})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_things_eeg2_test(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (eeg, times_ms). eeg: (N_images, n_reps, channels, T)."""
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()
    data = np.asarray(arr["preprocessed_eeg_data"], dtype=np.float32)
    times = np.asarray(arr["times"], dtype=float)
    if np.median(np.diff(times)) < 0.5:          # seconds -> ms
        times = times * 1000.0
    if data.ndim != 4:
        raise ValueError(f"expected (img, rep, ch, t), got {data.shape}")
    return data, times


def smooth_time(eeg: np.ndarray, width: int) -> np.ndarray:
    """Centered boxcar over the last axis (width in samples; 1 = none)."""
    if width <= 1:
        return eeg
    pad = width // 2
    x = np.pad(eeg, [(0, 0)] * (eeg.ndim - 1) + [(pad, width - 1 - pad)], mode="edge")
    c = np.cumsum(x, axis=-1)
    c = np.concatenate([np.zeros_like(c[..., :1]), c], axis=-1)
    return (c[..., width:] - c[..., :-width]) / width


# ---------------------------------------------------------------------------
# Cross-validated RDMs
# ---------------------------------------------------------------------------

def cv_rdm_series(eeg: np.ndarray, n_splits: int = 8, rng=None) -> np.ndarray:
    """Cross-validated squared-Euclidean RDM per time bin.

    eeg: (N, R, C, T). For a random split of reps into halves with means A, B:
        d_ij = (A_i - A_j) . (B_i - B_j) / C
    Noise is independent across halves, so E[d_ij] = true squared distance.
    Averaged over n_splits random splits. Returns (T, N, N).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n, r, c, t = eeg.shape
    out = np.zeros((t, n, n))
    for _ in range(n_splits):
        perm = rng.permutation(r)
        a = eeg[:, perm[: r // 2]].mean(1).transpose(2, 0, 1)   # (T, N, C)
        b = eeg[:, perm[r // 2:]].mean(1).transpose(2, 0, 1)
        m = np.einsum("tic,tjc->tij", a, b)
        dg = np.einsum("tii->ti", m)
        out += dg[:, :, None] + dg[:, None, :] - m - m.transpose(0, 2, 1)
    out /= n_splits * c
    idx = np.arange(n)
    out[:, idx, idx] = 0.0
    return out


def subject_rdms(subject: str, eeg: np.ndarray, times_ms: np.ndarray,
                 n_splits: int = 8, n_ceiling_splits: int = 10,
                 seed: int = 0) -> SubjectRDMs:
    rng = np.random.default_rng(seed)
    n, r, _c, t = eeg.shape
    full = cv_rdm_series(eeg, n_splits, rng)

    # fixed rep-halves for unbiased layer selection / evaluation
    perm = rng.permutation(r)
    ha, hb = perm[: r // 2], perm[r // 2:]
    half_a = cv_rdm_series(eeg[:, ha], n_splits, rng)
    half_b = cv_rdm_series(eeg[:, hb], n_splits, rng)

    # split-half reliability per bin, averaged over random half-splits
    rel = np.zeros(t)
    for s in range(n_ceiling_splits):
        if s == 0:
            ra, rb = half_a, half_b
        else:
            p = rng.permutation(r)
            ra = cv_rdm_series(eeg[:, p[: r // 2]], max(2, n_splits // 2), rng)
            rb = cv_rdm_series(eeg[:, p[r // 2:]], max(2, n_splits // 2), rng)
        rel += np.array([align.rsa(ra[i], rb[i]) for i in range(t)])
    rel /= n_ceiling_splits
    sb = 2 * rel / (1 + rel)                         # Spearman-Brown to full data
    ceiling = np.sqrt(np.clip(sb, 0.0, 1.0))
    return SubjectRDMs(subject, times_ms, full, half_a, half_b, sb, ceiling)


# ---------------------------------------------------------------------------
# Between-subject ceiling (Nili et al. 2014)
# ---------------------------------------------------------------------------

def group_ceiling(rdms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """rdms: list over subjects of (T, N, N). Returns (lower, upper) per bin."""
    s = len(rdms)
    if s < 2:
        raise ValueError("need >= 2 subjects for a between-subject ceiling")
    t = rdms[0].shape[0]
    lo, hi = np.zeros(t), np.zeros(t)
    for i in range(t):
        ranked = np.stack([align.rankdata(align.upper(r[i])) for r in rdms])
        mean_all = ranked.mean(0)
        for k in range(s):
            others = (mean_all * s - ranked[k]) / (s - 1)
            hi[i] += align.pearson(ranked[k], mean_all)
            lo[i] += align.pearson(ranked[k], others)
    return lo / s, hi / s


def build_subject(subject: str, eeg_path: str, out_dir: str, smooth: int = 1,
                  **kw) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"brain_{subject}.npz")
    if os.path.exists(dest):
        return dest
    eeg, times = load_things_eeg2_test(eeg_path)
    eeg = smooth_time(eeg, smooth)
    subject_rdms(subject, eeg, times, **kw).save(dest)
    return dest
