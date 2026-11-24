"""Brain-free proxies recover planted structure."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hrh import align, proxies as px  # noqa: E402


def planted(alpha, n=200, d=400, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.arange(1, d + 1, dtype=float) ** (-alpha / 2)
    x = rng.normal(size=(n, d)) * scale
    return x @ x.T


def test_spectral_alpha_recovers_planted_exponent():
    for a in (0.5, 1.0, 1.5):
        est = px.spectral_alpha(planted(a))
        assert abs(est - a) < 0.15, (a, est)
    assert px.spectral_alpha(planted(0.5)) < px.spectral_alpha(planted(1.5))


def test_gabor_features_separate_orientation():
    from PIL import Image
    yy, xx = np.mgrid[0:128, 0:128]
    paths = []
    with tempfile.TemporaryDirectory() as d:
        for i, th in enumerate([0, 0, np.pi / 2, np.pi / 2]):
            g = 0.5 + 0.5 * np.sin(2 * np.pi / 8 * (xx * np.cos(th) + yy * np.sin(th)) + i)
            p = os.path.join(d, f"g{i}.png")
            Image.fromarray((g * 255).astype(np.uint8)).save(p)
            paths.append(p)
        rdm = px.features_to_rdm(px.gabor_features(paths))
    assert rdm[0, 1] < rdm[0, 2] and rdm[2, 3] < rdm[1, 3]


def test_semantic_rdm_jaccard_and_mask():
    m = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 0]])
    d, valid = px.semantic_rdm(m)
    assert d[0, 1] == 0 and d[0, 2] == 1 and valid.tolist() == [True, True, True, False]


def test_layer_proxies_track_planted_retention():
    rng = np.random.default_rng(1)
    low, sem = rng.normal(size=(60, 20)), rng.normal(size=(60, 20))
    low_rdm, sem_rdm = px.features_to_rdm(low), px.features_to_rdm(sem)
    layers = [low, 0.5 * low + 0.5 * sem, sem]
    grams = np.stack([x @ x.T for x in layers])
    rdms = np.stack([align.gram_to_corr_rdm(x @ x.T, x.sum(1), x.shape[1]) for x in layers])
    p = px.layer_proxies(grams, rdms, low_rdm, sem_rdm, np.ones(60, bool))
    assert p["L"][0] > p["L"][1] > p["L"][2]
    assert p["S"][2] > p["S"][1] > p["S"][0]
