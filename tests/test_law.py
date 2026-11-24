"""The law recovers a planted relation, stays null on noise, and registration is enforced."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hrh import evaluate as ev  # noqa: E402
from hrh import law as lw  # noqa: E402


def world(planted=True, n_fam=6, per=5, seed=0):
    rng = np.random.default_rng(seed)
    prox, scores, params, family = {}, {}, {}, {}
    for f in range(n_fam):
        for k in range(per):
            m = f"f{f}_m{k}"
            alpha = np.sort(rng.uniform(0.2, 2.5, 8))
            px = {"L_max": rng.uniform(0.1, 0.6), "S_max": rng.uniform(0.1, 0.6),
                  "alpha": alpha.tolist()}
            prox[m], params[m], family[m] = px, 10 ** rng.uniform(1, 3), f"fam{f}"
            if planted:
                s = {w: 0.1 + 0.4 * px["L_max"] + 0.2 * px["S_max"]
                     - 0.3 * lw.spectral_mismatch(alpha, b) ** 2 + 0.01 * rng.normal()
                     for w, b in (("early", 1.6), ("mid", 1.0), ("late", 0.6))}
            else:
                s = {w: rng.normal() for w in lw.WINDOWS}
            scores[m] = s
    return prox, scores, params, family


def test_law_recovers_planted_relation():
    law = lw.fit(*world(True))
    for w, b in (("early", 1.6), ("mid", 1.0), ("late", 0.6)):
        spec = law["windows"][w]
        assert abs(spec["b"] - b) <= 0.15
        assert spec["cv"]["law"]["spearman"] > 0.8
        assert spec["cv"]["law"]["spearman"] > spec["cv"]["log_params"]["spearman"]


def test_law_is_null_on_noise():
    law = lw.fit(*world(False))
    assert all(abs(law["windows"][w]["cv"]["law"]["spearman"]) < 0.5 for w in lw.WINDOWS)


def test_verdict_rule():
    up = {"mean": 0.5, "ci95": [0.2, 0.8]}
    down = {"mean": 0.1, "ci95": [-0.3, 0.5]}
    pos = {"mean": 0.2}
    assert lw.law_verdict({"early": up, "late": up}, {"early": pos, "late": pos})["label"] == "supported"
    assert lw.law_verdict({"early": up, "late": down}, {"early": pos, "late": pos})["label"] == "partial"
    assert lw.law_verdict({"early": up, "late": up},
                          {"early": {"mean": -0.1}, "late": {"mean": -0.1}})["label"] == "not supported"


def test_registration_hash_is_enforced():
    with tempfile.TemporaryDirectory() as d:
        assert ev.registered_predictions(d) is None
        reg = {"predictions": {"m": {"early": 0.1}}}
        with open(os.path.join(d, "predictions_heldout.json"), "w") as fh:
            json.dump(reg, fh)
        with open(os.path.join(d, "predictions_heldout.sha256"), "w") as fh:
            fh.write(lw.sha256(reg) + "  predictions_heldout.json\n")
        assert ev.registered_predictions(d) == reg
        with open(os.path.join(d, "predictions_heldout.json"), "w") as fh:
            json.dump({"predictions": {"m": {"early": 0.9}}}, fh)
        with pytest.raises(SystemExit):
            ev.registered_predictions(d)
