"""Ground-truth tests for the HRH measurement core (numpy only).

Each measure must recover the planted trajectory property; the noise model and
the pre-stimulus window must come out at chance. If any test fails, no
real-data result from that measure should be trusted.

Run: pytest -q   (or: python3 tests/test_synthetic.py)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hrh import align, brain, synthetic as syn, trajectory as tr  # noqa: E402

N = 120


def _setup(seed=1):
    rng = np.random.default_rng(seed)
    f = syn.factors(rng, N)
    subj = brain.subject_rdms("synth", syn.eeg(rng, f), syn.TIMES,
                              n_splits=4, n_ceiling_splits=3)
    mask = tr.reliable_mask(syn.TIMES, subj.split_half_r)
    res = {}
    for kind in ("hier", "static", "noise"):
        rd = syn.model_rdms(rng, f, kind)
        a = tr.alignment_matrix(rd, subj.full)
        an = tr.normalize(a, subj.ceiling, mask)
        res[kind] = dict(cov=tr.coverage_auc(an), hier=tr.hierarchy_slope(a, syn.TIMES, mask),
                         dtw=tr.dtw_distance(an), peak=np.nanmax(an),
                         pre=np.abs(a[:, syn.TIMES < 0]).max(),
                         win=tr.window_scores(rd, subj))
    return subj, mask, res


_CACHE = {}


def setup():
    if not _CACHE:
        _CACHE["v"] = _setup()
    return _CACHE["v"]


def test_metric_invariances_and_nulls():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, 16))
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    sq = lambda m: ((m[:, None] - m[None]) ** 2).sum(-1)  # noqa: E731
    dx, dy, dz = sq(x), sq(x @ q), sq(rng.normal(size=(N, 16)))
    k = align.dist_to_kernel
    assert abs(align.cka_debiased(k(dx), k(dy)) - 1) < 1e-6        # rotation invariant
    assert align.mutual_knn(dx, dy) == 1.0
    assert abs(align.cka_debiased(k(dx), k(dz))) < 0.05             # debiased null ~ 0
    assert align.mutual_knn(dx, dz) < 0.2
    assert np.allclose(align.rankdata(np.array([3., 1, 2, 2, 5])), [4, 1, 2.5, 2.5, 5])


def test_ceiling_separates_signal_from_noise():
    subj, mask, _ = setup()
    t = syn.TIMES
    pre = subj.ceiling[t < 0].mean()
    sig = subj.ceiling[(t >= 60) & (t <= 400)].mean()
    assert sig > 3 * max(pre, 0.05)
    assert not mask[t > 550].any(), "pure-noise late bins must be excluded"


def test_platonic_vs_heraclitean_dissociation():
    _, _, r = setup()
    h, s = r["hier"], r["static"]
    assert s["win"]["mid"]["norm"] >= 0.8 * h["win"]["mid"]["norm"]   # static matches mid
    assert h["cov"] > s["cov"] + 0.1                                 # ...but not coverage
    assert h["dtw"] < s["dtw"]                                       # ...nor trajectory
    assert h["hier"] > 0.5
    assert h["win"]["early"]["norm"] > s["win"]["early"]["norm"] + 0.2
    assert h["win"]["late"]["norm"] > s["win"]["late"]["norm"] + 0.2


def test_nulls_at_chance():
    _, _, r = setup()
    assert r["noise"]["cov"] < 0.15 and r["noise"]["peak"] < 0.4
    for kind, v in r.items():
        assert v["pre"] < 0.1, f"{kind}: alignment before stimulus onset"


def test_family_slope():
    fam = {m: "F" for m in ("a", "b", "c")}
    par = {"a": 10, "b": 100, "c": 1000}
    assert abs(tr.family_slope({"a": .3, "b": .5, "c": .7}, par, fam) - 0.2) < 1e-9


def test_window_scores_other_metrics():
    subj, _, _ = setup()
    rng = np.random.default_rng(3)
    f = syn.factors(rng, N)
    for metric in ("cka", "mknn"):
        ws = tr.window_scores(syn.model_rdms(rng, f, "noise"), subj, metric=metric)
        assert all(abs(v["score"]) < 0.2 for v in ws.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    _, _, r = setup()
    for k, v in r.items():
        print(f"  {k:<6} peak {v['peak']:.2f} cov {v['cov']:.2f} H {v['hier']:+.2f} "
              f"D {v['dtw']:.2f} " + " ".join(f"{w}={x['norm']:.2f}" for w, x in v["win"].items()))
    print("ALL CHECKS PASSED")
