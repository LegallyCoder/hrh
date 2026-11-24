"""Synthetic ground truth with planted trajectories (used by tests and paper figures).

Brain: three latent regimes (low / mid / late) that peak at ~85, ~185, ~330 ms,
plus pure noise before 40 ms and after 450 ms.
Models:
  hier   : layers walk low -> mid -> late    (full trajectory, "Heraclitean")
  static : every layer is the mid factor     (single bottleneck, "Platonic")
  noise  : unrelated features                (null)
"""
from __future__ import annotations

import numpy as np

from . import align

TIMES = np.arange(-200, 800, 10.0)
REGIMES = ("low", "mid", "late")


def factors(rng, n=120, k=8):
    return {r: rng.normal(size=(n, k)) for r in REGIMES}


def regime_weights(t):
    t = np.asarray(t, float)
    g = lambda mu, sd: np.exp(-0.5 * ((t - mu) / sd) ** 2)  # noqa: E731
    w = np.array([g(85, 30), g(185, 45), g(330, 60)])
    w[:, (t < 40) | (t > 450)] = 0.0
    return w


def eeg(rng, f, reps=20, channels=32, noise=4.0, times=TIMES):
    n, k = f["low"].shape
    patt = {r: f[r] @ rng.normal(size=(k, channels)) for r in REGIMES}
    w = regime_weights(times)
    sig = sum(w[i][None, None, :] * patt[r][:, :, None] for i, r in enumerate(REGIMES))
    out = sig[:, None] + noise * rng.normal(size=(n, reps, channels, len(times)))
    return out.astype(np.float32)


def model_features(rng, f, kind, depth=10):
    n, k = f["low"].shape
    feats = []
    for i in range(depth):
        d = i / (depth - 1)
        if kind == "hier":
            w = regime_weights([40 + d * 400.0])[:, 0] + 1e-3
            x = np.concatenate([w[j] * f[r] for j, r in enumerate(REGIMES)], 1)
        elif kind == "static":
            x = f["mid"] + 0.05 * rng.normal(size=(n, k))
        elif kind == "noise":
            x = rng.normal(size=(n, 3 * k))
        else:
            raise ValueError(kind)
        feats.append(x)
    return feats


def model_rdms(rng, f, kind, depth=10):
    return np.stack([align.gram_to_corr_rdm(x @ x.T, x.sum(1), x.shape[1])
                     for x in model_features(rng, f, kind, depth)])
