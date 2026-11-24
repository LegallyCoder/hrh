"""Model-to-model convergence: planted readout convergence is detected, and only there."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hrh import convergence as cv  # noqa: E402
from hrh.features import LADDERS  # noqa: E402


def _world(d, planted, n=60, seed=0):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(n, 12))
    os.makedirs(os.path.join(d, "grams"))
    for fam in ("resnet-in1k", "vit-sup", "clip-laion"):
        lad = LADDERS[fam]
        for i, (m, p) in enumerate(lad):
            q = i / (len(lad) - 1)                      # 0 small .. 1 large
            layers = []
            for depth in range(5):
                own = rng.normal(size=(n, 12))
                if planted and depth == 4:              # readout: larger -> more shared
                    x = (0.3 + 1.5 * q) * shared + own
                else:
                    x = own + 0.3 * shared
                layers.append(x)
            g = np.stack([x @ x.T for x in layers]).astype(np.float32)
            s = np.stack([x.sum(1) for x in layers]).astype(np.float32)
            pay = {"layer_names": np.array([f"l{j}" for j in range(5)]), "params_m": p}
            for pool in ("grid4", "gap", "cls"):
                pay[f"gram_{pool}"], pay[f"sums_{pool}"] = g, s
                pay[f"dims_{pool}"], pay[f"ok_{pool}"] = np.full(5, 12), np.ones(5, bool)
            np.savez(os.path.join(d, "grams", m.replace(":", "_") + ".npz"), **pay)


def test_planted_readout_convergence():
    with tempfile.TemporaryDirectory() as d:
        _world(d, True)
        res = cv.run(d, n_sub=20)
    assert res["predictions"]["P1_prh_replicates_at_readout"]
    assert res["predictions"]["P2_convergence_is_readout_specific"]


def test_null_world():
    with tempfile.TemporaryDirectory() as d:
        _world(d, False)
        res = cv.run(d, n_sub=20)
    assert not res["predictions"]["P2_convergence_is_readout_specific"]
