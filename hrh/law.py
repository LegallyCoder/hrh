"""The matched-spectrum law: predict a model's brain alignment per window from
brain-free proxies (hrh/proxies.py), fit once on the development models, frozen,
then tested on held-out models whose predictions are registered in advance.

For window w the law is

    score_w = c0 + c1 * L_max + c2 * S_max - c3 * m_w^2,
    m_w     = min over layers |alpha(l) - b_w|

L_max / S_max: best low-level / semantic retention over layers; alpha(l): layer
spectral exponents; b_w: the window's preferred exponent (P4: alignment peaks
when spectra match; P5: the objective fixes which exponents a network visits).
b_w is chosen by leave-one-family-out cross-validation on a fixed grid, and c by
least squares. Baselines: log10(params) and a constant.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import align

B_GRID = np.round(np.arange(0.0, 3.01, 0.05), 2)
WINDOWS = ("early", "mid", "late")
FEATURES = ("const", "L_max", "S_max", "mismatch2")


def spectral_mismatch(alpha: list[float], b: float) -> float:
    a = np.asarray(alpha, float)
    a = a[np.isfinite(a)]
    return float(np.min(np.abs(a - b))) if len(a) else float("nan")


def design(px: dict, b: float) -> np.ndarray:
    return np.array([1.0, px["L_max"], px["S_max"], spectral_mismatch(px["alpha"], b) ** 2])


def _ols(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    return np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return align.pearson(align.rankdata(a[ok]), align.rankdata(b[ok]))


def _lofo(models, family, xfun, y):
    """Leave-one-family-out predictions for design function xfun(model)."""
    pred = {}
    for f in sorted(set(family[m] for m in models)):
        train = [m for m in models if family[m] != f]
        test = [m for m in models if family[m] == f]
        if len(train) < 4:
            continue
        coef = _ols(np.stack([xfun(m) for m in train]), np.array([y[m] for m in train]))
        for m in test:
            pred[m] = float(xfun(m) @ coef)
    return pred


def _cv_stats(pred: dict, y: dict) -> dict:
    ms = [m for m in pred if np.isfinite(pred[m])]
    p, o = np.array([pred[m] for m in ms]), np.array([y[m] for m in ms])
    ss = np.sum((o - o.mean()) ** 2)
    return {"spearman": _spearman(p, o), "r2": float(1 - np.sum((o - p) ** 2) / ss) if ss else None,
            "n": len(ms)}


def fit(proxies: dict, scores: dict, params: dict, family: dict) -> dict:
    """proxies: model -> summarize() dict; scores: model -> {window: score}.
    Returns the frozen law with cross-validated diagnostics and baselines."""
    models = sorted(m for m in proxies if m in scores)
    law = {"windows": {}, "n_models": len(models), "models": models, "b_grid": B_GRID.tolist(),
           "features": list(FEATURES)}
    for w in WINDOWS:
        y = {m: scores[m][w] for m in models}
        cv = {}
        for b in B_GRID:
            pred = _lofo(models, family, lambda m, b=b: design(proxies[m], b), y)
            o = np.array([y[m] for m in pred])
            p = np.array([pred[m] for m in pred])
            cv[float(b)] = float(np.mean((o - p) ** 2))
        b_best = min(cv, key=cv.get)
        x = np.stack([design(proxies[m], b_best) for m in models])
        coef = _ols(x, np.array([y[m] for m in models]))
        law_pred = _lofo(models, family, lambda m: design(proxies[m], b_best), y)
        par_pred = _lofo(models, family, lambda m: np.array([1.0, np.log10(params[m])]), y)
        const_pred = _lofo(models, family, lambda m: np.array([1.0]), y)
        law["windows"][w] = {
            "b": b_best, "coef": coef.round(6).tolist(),
            "cv": {"law": _cv_stats(law_pred, y), "log_params": _cv_stats(par_pred, y),
                   "constant": _cv_stats(const_pred, y)},
            "params_coef": _ols(np.stack([[1.0, np.log10(params[m])] for m in models]),
                                np.array([y[m] for m in models])).round(6).tolist(),
        }
    return law


def predict(law: dict, proxies: dict, params: dict) -> dict:
    out = {}
    for m, px in proxies.items():
        out[m] = {}
        for w, spec in law["windows"].items():
            out[m][w] = float(design(px, spec["b"]) @ np.array(spec["coef"]))
            out[m][f"{w}_params_baseline"] = float(
                np.array([1.0, np.log10(params[m])]) @ np.array(spec["params_coef"]))
    return out


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def law_verdict(rho: dict, diff: dict) -> dict:
    """Registered rule: supported iff the lower 95% bound of the Spearman correlation
    between predicted and observed held-out scores is > 0 in the early AND late
    windows, and the law's correlation exceeds the log-params baseline's in both
    (positive mean paired difference)."""
    ok = {w: (rho[w]["ci95"] or [np.nan])[0] > 0 and (diff[w]["mean"] or 0) > 0
          for w in ("early", "late")}
    label = "supported" if all(ok.values()) else "partial" if any(ok.values()) else "not supported"
    return {"label": label, **{f"{w}_ok": bool(v) for w, v in ok.items()}}
