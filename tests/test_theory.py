"""Numerical checks of the propositions in the paper's theory section."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hrh import theory as th  # noqa: E402

_C = {}


def runs():
    if not _C:
        prob = th.make_problem()
        _C["prob"] = prob
        _C["single"] = th.train(prob, steps=15000)
        _C["multi"] = th.train(prob, steps=15000, recon=1.0)
    return _C["prob"], _C["single"], _C["multi"]


def test_p1_nuisance_erasure():
    _, s, _ = runs()
    assert s["nuisance_l1"][0] > 0.5            # random init carries nuisance
    assert s["nuisance_l1"][-1] < 1e-3          # stationary point erases it


def test_p2_power_collapse():
    _, s, _ = runs()
    assert s["collapse_err"][-1] < 1e-2


def test_t1_trajectory_ceiling():
    prob, s, _ = runs()
    gs = th.grams(s["ws"])
    c = s["ceiling_early"]
    assert all(th.cos(g, prob["G_early"]) <= c + 1e-6 for g in gs)
    assert s["cka_early_l1"][0] > c             # an unoptimized net can exceed it


def test_p3_reconstruction_readout_removes_ceiling():
    _, s, m = runs()
    assert m["cka_early_l1"][-1] > s["ceiling_early"] + 0.1
    assert m["cka_mid_top"][-1] > 0.9 and s["cka_mid_top"][-1] > 0.9   # both solve the task


def test_balancedness_identity():
    _, s, _ = runs()
    ws = s["ws"]
    for a, b in zip(ws[:-1], ws[1:]):
        lhs, rhs = b.T @ b, a @ a.T
        assert np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs) < 1e-2


_A = {}


def aniso():
    if not _A:
        sig = th.make_spectrum(d=16, decay=1.0)
        lat = th.psd_power(sig, -0.5)                        # whitened latent target
        _A["sig"] = sig
        for name, tgt in (("pixel", np.eye(16)), ("latent", lat)):
            for depth in (4, 8):
                _A[name, depth] = th.train_aniso(sig, tgt, depth=depth,
                                                 lr=0.02 if depth == 4 else 0.01)
        _A["latent_w64"] = th.train_aniso(sig, lat, depth=4, width=64)
    return _A


def test_p4_spectral_matching_is_unimodal():
    eigs = np.arange(1, 33, dtype=float) ** -1.0
    for b in (0.0, 0.4, 1.0):
        grid = np.linspace(-0.5, 1.5, 81)
        c = np.array([th.spectral_cka(eigs, a, b) for a in grid])
        k = int(np.argmax(c))
        assert abs(grid[k] - b) < 0.03 and c[k] > 1 - 1e-9
        assert np.all(np.diff(c[:k + 1]) >= -1e-12) and np.all(np.diff(c[k:]) <= 1e-12)


def test_p5_spectral_trajectory():
    a = aniso()
    sig = a["sig"]
    ex = lambda g: th.spectral_exponent(g, 1, 12)          # noqa: E731
    for depth in (4, 8):
        pix = [ex(g) for g in th.whitened_grams(a["pixel", depth], sig)]
        lat = [ex(g) for g in th.whitened_grams(a["latent", depth], sig)]
        want = 1 - np.arange(1, depth + 1) / depth
        assert np.allclose(pix, 1.0, atol=0.03)            # reconstruction keeps the input spectrum
        assert np.allclose(lat, want, atol=0.05)           # latent target sweeps [0, 1]
    w64 = [ex(g) for g in th.whitened_grams(a["latent_w64"], sig)]
    assert np.allclose(w64, 1 - np.arange(1, 5) / 4, atol=0.05)   # width does not enter


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("  ok ", n)
    prob, s, m = runs()
    print(f"  ceiling rho_S = {s['ceiling_early']:.3f}; CKA(l1, early): init "
          f"{s['cka_early_l1'][0]:.3f} -> single {s['cka_early_l1'][-1]:.3f} | "
          f"multi {m['cka_early_l1'][-1]:.3f}")
