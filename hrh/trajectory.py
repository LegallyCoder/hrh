"""HRH quantities: everything that turns 'alignment' into a function of time.

    A[l, t]        layer x brain-time alignment matrix (RSA / CKA / mkNN)
    A_norm         A divided by the per-bin noise ceiling
    reliable bins  bins whose ceiling exceeds every PRE-STIMULUS bin's ceiling
                   (data-driven: no hand-picked threshold, so late bins that are
                   pure noise cannot count against a model)
    coverage       C(tau) = frac. of reliable bins where max_l A_norm >= tau;
                   coverage_auc = integral_0^1 C(tau) dtau = mean clip(max_l A_norm)
    hierarchy      Spearman(t, argmax_l A[:, t]) over reliable bins: does model
                   depth track brain time?
    D_HRH          DTW trajectory distance on 1 - clip(A_norm): can the WHOLE
                   layer sequence be monotonically matched to the WHOLE brain
                   timeline? (high peak + bad D_HRH = "Platonic but not Heraclitean")
    window scores  per window, best layer chosen on rep-half A, scored on
                   rep-half B (and vice versa): no selection bias
    scaling slope  within-family OLS slope of window score on log10(params),
                   family fixed effects, hierarchical bootstrap (subjects x stimuli)
"""
from __future__ import annotations

import numpy as np

from . import align

WINDOWS = {"early": (50, 120), "mid": (120, 250), "late": (250, 600)}


# ---------------------------------------------------------------------------
# A matrix
# ---------------------------------------------------------------------------

def _zranked(rdms: np.ndarray) -> np.ndarray:
    z = np.stack([align.rankdata(align.upper(r)) for r in rdms])
    z -= z.mean(1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    return z


def alignment_matrix(model_rdms: np.ndarray, brain_rdms: np.ndarray,
                     metric: str = "rsa", k: int = 10) -> np.ndarray:
    """(L, N, N) x (T, N, N) -> (L, T)."""
    if metric == "rsa":
        return _zranked(model_rdms) @ _zranked(brain_rdms).T
    L, T = len(model_rdms), len(brain_rdms)
    out = np.zeros((L, T))
    if metric == "cka":
        km = [align.dist_to_kernel(r) for r in model_rdms]
        kb = [align.dist_to_kernel(r) for r in brain_rdms]
        for i in range(L):
            for j in range(T):
                out[i, j] = align.cka_debiased(km[i], kb[j])
    elif metric == "mknn":
        nm = [align.knn_sets(r, k) for r in model_rdms]
        nb = [align.knn_sets(r, k) for r in brain_rdms]
        n = model_rdms.shape[1]
        rows = np.arange(n)[:, None]

        def mask(s):
            m = np.zeros((n, n), bool)
            m[rows, s] = True
            return m
        mm, mb = [mask(s) for s in nm], [mask(s) for s in nb]
        for i in range(L):
            for j in range(T):
                out[i, j] = (mm[i] & mb[j]).sum(1).mean() / k
    else:
        raise ValueError(metric)
    return out


def reliable_mask(times_ms: np.ndarray, reliability: np.ndarray,
                  alpha: float = 0.05, min_run: int = 3) -> np.ndarray:
    """Post-onset bins whose split-half reliability is significantly above the
    pre-stimulus null, Bonferroni-corrected over post-onset bins, kept only in
    runs of >= min_run contiguous bins (cluster rule, as is standard for EEG).
    `reliability` is the per-bin split-half r (not the sqrt ceiling, which
    inflates small values)."""
    from statistics import NormalDist
    base = reliability[times_ms < 0]
    post = times_ms >= 0
    mu, sd = (base.mean(), base.std(ddof=1)) if len(base) > 2 else (0.0, 0.05)
    z = NormalDist().inv_cdf(1 - alpha / max(post.sum(), 1))
    sig = post & (reliability > mu + z * max(sd, 1e-3))
    out = np.zeros_like(sig)
    i = 0
    while i < len(sig):
        if sig[i]:
            j = i
            while j < len(sig) and sig[j]:
                j += 1
            if j - i >= min_run:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def normalize(a: np.ndarray, ceiling: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = a / np.where(ceiling > 0, ceiling, np.nan)[None, :]
    out[:, ~mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# Trajectory descriptors
# ---------------------------------------------------------------------------

def _best_per_bin(a_norm: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a_norm).any(0)
    return np.nanmax(a_norm[:, ok], axis=0) if ok.any() else np.zeros(0)


def coverage_curve(a_norm: np.ndarray, taus=np.linspace(0, 1, 21)) -> np.ndarray:
    best = _best_per_bin(a_norm)
    return np.array([(best >= t).mean() if len(best) else np.nan for t in taus])


def coverage_auc(a_norm: np.ndarray) -> float:
    best = _best_per_bin(a_norm)
    return float(np.clip(best, 0, 1).mean()) if len(best) else float("nan")


def hierarchy_slope(a: np.ndarray, times_ms: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 3:
        return float("nan")
    depth = np.argmax(a[:, mask], axis=0) / max(len(a) - 1, 1)
    return align.pearson(align.rankdata(times_ms[mask]), align.rankdata(depth))


def subsample_depth(a: np.ndarray, n_depth: int) -> np.ndarray:
    """Equalize layer count across models (max-over-layers is biased toward deep
    nets with many blocks; this removes that multiple-comparisons advantage)."""
    if n_depth is None or len(a) <= n_depth:
        return a
    idx = np.round(np.linspace(0, len(a) - 1, n_depth)).astype(int)
    return a[idx]


def dtw_distance(a_norm: np.ndarray) -> float:
    """Symmetric DTW (Sakoe-Chiba weights: diagonal 2, horizontal/vertical 1)
    from (first layer, first reliable bin) to (last layer, last reliable bin),
    normalized by L + T, so the result is the weighted mean cost in [0, 1]."""
    cols = np.isfinite(a_norm).all(0)
    cost = 1.0 - np.clip(a_norm[:, cols], 0, 1)
    L, T = cost.shape
    if L == 0 or T == 0:
        return float("nan")
    acc = np.full((L + 1, T + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, L + 1):
        for j in range(1, T + 1):
            c = cost[i - 1, j - 1]
            acc[i, j] = min(acc[i - 1, j - 1] + 2 * c, acc[i - 1, j] + c, acc[i, j - 1] + c)
    return float(acc[L, T] / (L + T))


# ---------------------------------------------------------------------------
# Window scores (cross-validated layer selection)
# ---------------------------------------------------------------------------

def window_rdm(rdms: np.ndarray, times_ms: np.ndarray, win) -> np.ndarray:
    sel = (times_ms >= win[0]) & (times_ms < win[1])
    return rdms[sel].mean(0)


def window_scores(model_rdms: np.ndarray, subj, windows=WINDOWS,
                  metric: str = "rsa") -> dict:
    """Returns {window: dict(score, norm, layer_a, layer_b, ceiling)}.

    The ceiling is sqrt(metric(half A, half B)), which is exact for RSA under
    signal + independent noise and approximate for CKA / mkNN. Within one
    subject it is the same for every model, so within-family slope SIGNS do not
    depend on it."""
    out = {}
    for w, win in windows.items():
        ra = window_rdm(subj.half_a, subj.times_ms, win)
        rb = window_rdm(subj.half_b, subj.times_ms, win)
        sa = alignment_matrix(model_rdms, ra[None], metric)[:, 0]
        sb = alignment_matrix(model_rdms, rb[None], metric)[:, 0]
        la, lb = int(np.argmax(sa)), int(np.argmax(sb))
        score = 0.5 * (sb[la] + sa[lb])                   # select on one, test on other
        self_ab = float(alignment_matrix(ra[None], rb[None], metric)[0, 0])
        ceil = np.sqrt(max(self_ab, 1e-6))
        out[w] = {"score": float(score), "norm": float(score / ceil),
                  "layer_a": la, "layer_b": lb, "ceiling": ceil}
    return out


# ---------------------------------------------------------------------------
# Scaling slope + hierarchical bootstrap
# ---------------------------------------------------------------------------

def family_slope(scores: dict[str, float], params: dict[str, float],
                 family: dict[str, str]) -> float:
    """OLS slope of score on log10(params) with family fixed effects
    (demean x and y within family; families with one model drop out)."""
    xs, ys = [], []
    fams: dict[str, list[str]] = {}
    for m in scores:
        fams.setdefault(family[m], []).append(m)
    for ms in fams.values():
        ms = [m for m in ms if np.isfinite(scores[m])]
        if len(ms) < 2:
            continue
        x = np.log10([params[m] for m in ms])
        y = np.array([scores[m] for m in ms])
        xs.append(x - x.mean())
        ys.append(y - y.mean())
    if not xs:
        return float("nan")
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float((x @ y) / (x @ x)) if x @ x > 0 else float("nan")


def bootstrap_window_stats(model_rdms: dict[str, np.ndarray], subjects: list,
                           selected: dict, stats: dict, windows=WINDOWS,
                           n_boot: int = 1000, seed: int = 0) -> dict:
    """Hierarchical bootstrap (subjects, then stimuli; one shared stimulus draw
    because every subject saw the same images). Layers stay fixed at their
    cross-validated selection (`selected[subject][model][window] = (la, lb)`).

    stats: {name: f(scores)} where scores = {model: subject-mean RSA in the window};
    a statistic with attribute window_aware=True is called as f(scores, window).
    Every statistic is evaluated on the same draws, so contrasts between them are
    paired. Returns {name: {window: array(n_boot)}}."""
    rng = np.random.default_rng(seed)
    n = next(iter(model_rdms.values())).shape[1]
    win_rdms = {s.subject: {w: (window_rdm(s.half_a, s.times_ms, win),
                                window_rdm(s.half_b, s.times_ms, win))
                            for w, win in windows.items()} for s in subjects}
    res = {k: {w: np.zeros(n_boot) for w in windows} for k in stats}
    iu, ju = np.triu_indices(n, 1)
    for b in range(n_boot):
        subs = [subjects[i] for i in rng.integers(0, len(subjects), len(subjects))]
        idx = rng.integers(0, n, n)
        keep = idx[iu] != idx[ju]                          # drop duplicate-stimulus pairs
        pi, pj = idx[iu][keep], idx[ju][keep]
        cache: dict = {}

        def vec(key, r):
            if key not in cache:
                v = align.rankdata(r[pi, pj])
                v -= v.mean()
                cache[key] = v / (np.linalg.norm(v) + 1e-12)
            return cache[key]
        for w in windows:
            scores = {}
            for m, rd in model_rdms.items():
                vals = []
                for s in subs:
                    la, lb = selected[s.subject][m][w]
                    ra, rb = win_rdms[s.subject][w]
                    va, vb = vec((s.subject, w, "a"), ra), vec((s.subject, w, "b"), rb)
                    vals.append(0.5 * (vec((m, la), rd[la]) @ vb
                                       + vec((m, lb), rd[lb]) @ va))
                scores[m] = float(np.mean(vals))
            for k, f in stats.items():
                res[k][w][b] = f(scores, w) if getattr(f, "window_aware", False) else f(scores)
    return res


def bootstrap_window_slopes(model_rdms: dict[str, np.ndarray], subjects: list,
                            selected: dict, params: dict, family: dict,
                            windows=WINDOWS, n_boot: int = 1000,
                            seed: int = 0) -> dict:
    """Within-family scaling slope per window. Returns {window: array(n_boot)}."""
    out = bootstrap_window_stats(
        model_rdms, subjects, selected,
        {"slope": lambda sc: family_slope(sc, params, family)}, windows, n_boot, seed)
    return out["slope"]


def objective_contrast(scores: dict[str, float], members: list[str],
                       size_class: dict, references: dict) -> float:
    """Mean over `members` of [score - mean score of its size-matched references].
    Members without a scored reference drop out."""
    diffs = []
    for m in members:
        if m not in scores or not np.isfinite(scores[m]):
            continue
        refs = [scores[r] for r in references.get(size_class.get(m), []) if r in scores]
        if refs:
            diffs.append(scores[m] - float(np.mean(refs)))
    return float(np.mean(diffs)) if diffs else float("nan")
