"""PRH on its own terms: convergence BETWEEN models, by depth, with PRH's metric.

For every pair of models, mutual k-nearest-neighbour alignment (Huh et al., 2024)
between their layers at relative depths DEPTHS (1.0 = the readout embedding).
A model's convergence at depth d is its mean alignment with models of OTHER families.

Pre-specified predictions (written before this analysis was run):
  P1  PRH replicates at the readout: the within-family slope of readout convergence
      on log10(params) is positive (lower 95% bound > 0).
  P2  Convergence is a readout phenomenon: the readout slope exceeds the mid-depth
      (d = 0.5) slope (lower 95% bound of the difference > 0).
Exploratory: rank correlation of readout convergence with brain alignment per window.

Intervals: 100 random subsamples of 80% of the stimuli (160 of 200).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from . import align
from . import trajectory as tr
from .features import FAMILY, PARAMS, PRIMARY, model_rdms

DEPTHS = (0.25, 0.5, 0.75, 1.0)
K = 10
N_SUB, SUB_FRAC = 100, 0.8


def _onehot_knn(rdm: np.ndarray, k: int = K) -> np.ndarray:
    nn = align.knn_sets(rdm, k)
    m = np.zeros(rdm.shape, bool)
    m[np.arange(len(rdm))[:, None], nn] = True
    return m


def _pairwise(onehots: list[np.ndarray], k: int = K) -> np.ndarray:
    x = np.stack([o.ravel() for o in onehots]).astype(np.float32)   # (M, n*n)
    n = onehots[0].shape[0]
    return (x @ x.T) / (n * k)


def convergence_matrices(rdms: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict:
    """rdms: model -> (L, N, N). Returns {depth: (M, M) mutual-kNN matrix}."""
    models = sorted(rdms)
    out = {}
    for d in DEPTHS:
        oh = []
        for m in models:
            r = rdms[m][int(round(d * (len(rdms[m]) - 1)))]
            if idx is not None:
                r = r[np.ix_(idx, idx)]
            oh.append(_onehot_knn(r))
        out[d] = _pairwise(oh)
    return out


def centrality(mat: np.ndarray, models: list[str]) -> dict:
    fam = np.array([FAMILY[m] for m in models])
    out = {}
    for i, m in enumerate(models):
        other = fam != fam[i]
        out[m] = float(mat[i, other].mean())
    return out


def _slope(cent: dict, keep: set) -> float:
    sc = {m: v for m, v in cent.items() if m in keep}
    return tr.family_slope(sc, PARAMS, FAMILY)


def run(results_dir: str, pool: str = "grid4", seed: int = 0, n_sub: int = N_SUB) -> dict:
    rdms = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "grams", "*.npz"))):
        stem = os.path.basename(p)[:-4]
        name = next((m for m in PARAMS if m.replace(":", "_").replace("/", "_") == stem), None)
        if name is None:
            continue
        r, _n, _p = model_rdms(p, pool)
        if len(r) >= 2:
            rdms[name] = r
    models = sorted(rdms)
    ladders_all = {m for m in models if sum(FAMILY[x] == FAMILY[m] for x in models) >= 2}
    primary = {m for m in models if m in PRIMARY}

    def stats(mats):
        cent = {d: centrality(mats[d], models) for d in DEPTHS}
        return {"slope_primary": {d: _slope(cent[d], primary) for d in DEPTHS},
                "slope_all": {d: _slope(cent[d], ladders_all) for d in DEPTHS}, "cent": cent}

    full_m = convergence_matrices(rdms)
    full = stats(full_m)
    rng = np.random.default_rng(seed)
    n = next(iter(rdms.values())).shape[1]
    boots = []
    for _ in range(n_sub):
        idx = np.sort(rng.choice(n, int(SUB_FRAC * n), replace=False))
        boots.append(stats(convergence_matrices(rdms, idx)))

    def summ(key, d):
        b = np.array([x[key][d] for x in boots])
        return {"mean": full[key][d], "ci95": [float(np.percentile(b, 2.5)),
                                                float(np.percentile(b, 97.5))]}
    res = {"models": models, "depths": list(DEPTHS), "k": K,
           "matrices": {str(d): full_m[d].round(4).tolist() for d in DEPTHS},
           "centrality": {str(d): full["cent"][d] for d in DEPTHS},
           "slope_primary": {str(d): summ("slope_primary", d) for d in DEPTHS},
           "slope_all": {str(d): summ("slope_all", d) for d in DEPTHS}}
    diff = np.array([x["slope_primary"][1.0] - x["slope_primary"][0.5] for x in boots])
    res["readout_minus_mid"] = {"mean": full["slope_primary"][1.0] - full["slope_primary"][0.5],
                                "ci95": [float(np.percentile(diff, 2.5)),
                                         float(np.percentile(diff, 97.5))]}
    res["predictions"] = {
        "P1_prh_replicates_at_readout": bool(res["slope_primary"]["1.0"]["ci95"][0] > 0),
        "P2_convergence_is_readout_specific": bool(res["readout_minus_mid"]["ci95"][0] > 0)}
    csv_path = os.path.join(results_dir, f"models_{pool}.csv")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path) as fh:
            rows = {r["model"]: r for r in csv.DictReader(fh)}
        res["brain_link"] = {}
        for w in ("early", "mid", "late"):
            ms = [m for m in models if m in rows]
            res["brain_link"][w] = align.pearson(
                align.rankdata(np.array([full["cent"][1.0][m] for m in ms])),
                align.rankdata(np.array([float(rows[m][f"{w}_norm"]) for m in ms])))
    with open(os.path.join(results_dir, f"convergence_{pool}.json"), "w") as fh:
        json.dump(res, fh)
    return res
