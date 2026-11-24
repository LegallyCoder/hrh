"""Paper assets from the pipeline outputs: figures, tables and LaTeX macros.

Inputs (all small, written by `run_hrh.py eval|proxies|law|predict`):
  summary_grid4.json   registered tests (scaling, law_test) and slopes
  models_grid4.csv     per-model window scores (development + held-out)
  proxies_grid4.json   brain-free proxies
  registration/predictions_heldout.json

Everything labelled "exploratory" here was computed after the registered test.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from . import law as lw
from . import paperfigs as pf
from .features import FAMILY, HELDOUT, PARAMS

WIN = ("early", "mid", "late")


def _rows(path: str) -> list[dict]:
    with open(path) as fh:
        out = []
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            out.append(r)
    return out


def _model_boot_rho(pred, obs, n=4000, seed=0):
    """Bootstrap over models only (for exploratory subsets)."""
    rng = np.random.default_rng(seed)
    p, o = np.asarray(pred), np.asarray(obs)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(p), len(p))
        r = lw._spearman(p[i], o[i])
        if np.isfinite(r):
            vals.append(r)
    return float(lw._spearman(p, o)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def variance_shares(rows: list[dict]) -> dict:
    """Share of between-model variance explained by family (objective + architecture)
    and, within families, by log10(params)."""
    out = {}
    for w in WIN:
        y = np.array([r[f"{w}_norm"] for r in rows])
        fam = [r["family"] for r in rows]
        mu = {f: np.mean([yy for yy, ff in zip(y, fam) if ff == f]) for f in set(fam)}
        resid = y - np.array([mu[f] for f in fam])
        ss_tot = np.sum((y - y.mean()) ** 2)
        x = np.log10([r["params_m"] for r in rows])
        xm = {f: np.mean([xx for xx, ff in zip(x, fam) if ff == f]) for f in set(fam)}
        xc = x - np.array([xm[f] for f in fam])
        beta = (xc @ resid) / (xc @ xc)
        out[w] = {"family": float(1 - np.sum(resid ** 2) / ss_tot),
                  "size_within": float(np.sum((beta * xc) ** 2) / ss_tot)}
    return out


def _convergence_numbers(conv: dict) -> dict:
    models = conv["models"]
    cls = np.array([pf.FAMILY_CLASS.get(FAMILY.get(m), "supervised") for m in models])
    same = cls[:, None] == cls[None, :]
    iu = np.triu_indices(len(models), 1)
    out = {"agree_all": {}, "agree_same": {}, "agree_diff": {}}
    for d in conv["depths"]:
        mat = np.array(conv["matrices"][str(d)])[iu]
        out["agree_all"][d] = float(mat.mean())
        out["agree_same"][d] = float(mat[same[iu]].mean())
        out["agree_diff"][d] = float(mat[~same[iu]].mean())
    out["slope"] = conv["slope_primary"]
    out["brain"] = conv.get("brain_link", {})
    out["predictions"] = conv["predictions"]
    return out


def build(results_dir: str, paper_dir: str, reg_path: str) -> dict:
    gen, figs = os.path.join(paper_dir, "generated"), os.path.join(paper_dir, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    with open(os.path.join(results_dir, "summary_grid4.json")) as fh:
        summ = json.load(fh)
    with open(os.path.join(results_dir, "proxies_grid4.json")) as fh:
        prox = json.load(fh)
    with open(reg_path) as fh:
        reg = json.load(fh)
    law, pred = reg["law"], reg["predictions"]
    rows = _rows(os.path.join(results_dir, "models_grid4.csv"))
    obs = {r["model"]: {w: r[f"{w}_norm"] for w in WIN} for r in rows}
    dev_m = sorted(m for m in law["models"] if m in obs and m in prox)   # the registered fit set
    held_m = sorted(m for m in HELDOUT if m in obs and m in pred)

    # development: cross-validated predictions of the frozen procedure
    dev = {}
    for w in WIN:
        b = law["windows"][w]["b"]
        y = {m: obs[m][w] for m in dev_m}
        cvp = lw._lofo(dev_m, FAMILY, lambda m, b=b: lw.design(prox[m], b), y)
        for m, p in cvp.items():
            dev.setdefault(m, {})[w] = (p, y[m])
    held = {m: {w: (pred[m][w], obs[m][w]) for w in WIN} for m in held_m}

    # exploratory robustness
    explo = {"without_fcmae": {}, "ablation": {}, "variance": variance_shares(rows)}
    keep = [m for m in held_m if FAMILY[m] != "fcmae"]
    for w in WIN:
        r, ci = _model_boot_rho([pred[m][w] for m in keep], [obs[m][w] for m in keep])
        explo["without_fcmae"][w] = {"rho": r, "ci95": ci, "n": len(keep)}
    scores = {m: obs[m] for m in dev_m}
    for name, feats in lw.ABLATIONS.items():
        explo["ablation"][name] = {}
        for w in WIN:
            spec = lw.fit_subset({m: prox[m] for m in dev_m}, scores, FAMILY, feats, w)
            hp = [lw.predict_subset(spec, prox[m]) for m in held_m]
            if name == "full law":                  # the registered predictions themselves
                hp = [pred[m][w] for m in held_m]
            explo["ablation"][name][w] = {"dev_cv": spec["cv_spearman"],
                                          "heldout": lw._spearman(hp, [obs[m][w] for m in held_m])}

    # figures: simple main-text versions, detailed versions for the appendix
    bw = {w: law["windows"][w]["b"] for w in WIN}
    pf.fig_spectral(os.path.join(figs, "theory2.pdf"), b_windows=bw)
    pf.fig_depth(prox, FAMILY, os.path.join(figs, "depth.pdf"))
    pf.fig_scale_simple(rows, summ["slope_vs_log10_params"], explo["variance"],
                        os.path.join(figs, "scale.pdf"))
    pf.fig_law_simple(held, FAMILY, summ["law_test"], os.path.join(figs, "law.pdf"))
    pf.fig_measured(prox, FAMILY, bw, os.path.join(figs, "spectral_measured.pdf"))
    pf.fig_scale_objective(rows, summ["slope_vs_log10_params"],
                           os.path.join(figs, "scale_windows.pdf"), set(held_m))
    extremes = {}
    for w in WIN:
        srt = sorted(held_m, key=lambda m: obs[m][w])
        extremes[w] = [srt[0], srt[-1]]
    pf.fig_law(dev, held, FAMILY, summ.get("law_test"),
               {w: law["windows"][w]["cv"] for w in WIN},
               os.path.join(figs, "law_full.pdf"), extremes)
    pf.fig_ablation(explo, os.path.join(figs, "ablation.pdf"))
    conv_path = os.path.join(results_dir, "convergence_grid4.json")
    if os.path.exists(conv_path):
        with open(conv_path) as fh:
            conv = json.load(fh)
        pf.fig_convergence(conv, FAMILY, rows, os.path.join(figs, "convergence.pdf"))
        explo["convergence"] = _convergence_numbers(conv)
    amat = os.path.join(results_dir, "A_matrices_grid4.npz")
    if os.path.exists(amat):
        pf.fig_time(amat, FAMILY, os.path.join(figs, "time.pdf"),
                    {"eva02_large_patch14_224.mim_in22k": "EVA-02-L",
                     "vit_large_patch14_clip_224.laion2b": "CLIP-L/14",
                     "convnextv2_large.fcmae": "ConvNeXt V2-L"})

    # macros
    lt = summ.get("law_test", {})

    def ci(d):
        if not d or d.get("ci95") is None:
            return r"\tbd"
        return f"${d['mean']:+.2f}$ [${d['ci95'][0]:+.2f}$, ${d['ci95'][1]:+.2f}$]"
    def num3(x):
        t = f"{x:+.3f}"
        return "0.000" if t in ("+0.000", "-0.000") else t

    def ci3(d):
        return f"${num3(d['mean'])}$ [${num3(d['ci95'][0])}$, ${num3(d['ci95'][1])}$]"
    m = [r"% AUTO-GENERATED by `run_hrh.py paper` -- do not edit by hand.", r"\hrhvtwotrue",
         rf"\renewcommand{{\vNdev}}{{{len(dev_m)}}}", rf"\renewcommand{{\vNheld}}{{{len(held_m)}}}",
         rf"\renewcommand{{\vNall}}{{{len(rows)}}}",
         rf"\renewcommand{{\vLawVerdict}}{{{(lt.get('verdict') or {}).get('label', 'pending')}}}",
         rf"\renewcommand{{\vHash}}{{\texttt{{{lw.sha256(reg)[:16]}}}}}"]
    names = {"early": "Early", "mid": "Mid", "late": "Late"}
    for w in WIN:
        W = names[w]
        cvw = law["windows"][w]["cv"]
        m += [rf"\renewcommand{{\vRho{W}}}{{{ci((lt.get('law_rho') or {}).get(w))}}}",
              rf"\renewcommand{{\vPar{W}}}{{{ci((lt.get('params_rho') or {}).get(w))}}}",
              rf"\renewcommand{{\vCv{W}}}{{${cvw['law']['spearman']:+.2f}$}}",
              rf"\renewcommand{{\vCvR{W}}}{{${cvw['law']['r2']:.2f}$}}",
              rf"\renewcommand{{\vB{W}}}{{{law['windows'][w]['b']:.2f}}}",
              rf"\renewcommand{{\vNoFc{W}}}{{${explo['without_fcmae'][w]['rho']:+.2f}$}}",
              rf"\renewcommand{{\vVarFam{W}}}{{{100 * explo['variance'][w]['family']:.0f}}}",
              rf"\renewcommand{{\vVarSize{W}}}{{{100 * explo['variance'][w]['size_within']:.1f}}}",
              rf"\renewcommand{{\vSlope{W}}}{{{ci3(summ['slope_vs_log10_params'][w])}}}"]
    cv_ = explo.get("convergence")
    if cv_:
        def s4(d):
            x = cv_["slope"][str(d)]
            return f"${x['mean']:+.3f}$ [${x['ci95'][0]:+.3f}$, ${x['ci95'][1]:+.3f}$]"
        m += [r"\hrhconvtrue",
              rf"\renewcommand{{\vConvSlopeOut}}{{{s4(1.0)}}}",
              rf"\renewcommand{{\vConvSlopeMid}}{{{s4(0.5)}}}",
              rf"\renewcommand{{\vAgreeQ}}{{{cv_['agree_all'][0.25]:.2f}}}",
              rf"\renewcommand{{\vAgreeOut}}{{{cv_['agree_all'][1.0]:.2f}}}",
              rf"\renewcommand{{\vAgreeSameTQ}}{{{cv_['agree_same'][0.75]:.2f}}}",
              rf"\renewcommand{{\vAgreeSameOut}}{{{cv_['agree_same'][1.0]:.2f}}}",
              rf"\renewcommand{{\vAgreeDiffQ}}{{{cv_['agree_diff'][0.25]:.2f}}}",
              rf"\renewcommand{{\vAgreeDiffOut}}{{{cv_['agree_diff'][1.0]:.2f}}}",
              rf"\renewcommand{{\vConvBrainEarly}}{{{cv_['brain']['early']:.2f}}}",
              rf"\renewcommand{{\vConvBrainMid}}{{{cv_['brain']['mid']:.2f}}}",
              rf"\renewcommand{{\vConvBrainLate}}{{{cv_['brain']['late']:.2f}}}"]
    fams = [100 * explo["variance"][w]["family"] for w in WIN]
    m += [rf"\renewcommand{{\vVarFamMin}}{{{min(fams):.0f}}}",
          rf"\renewcommand{{\vVarFamMax}}{{{max(fams):.0f}}}"]
    bound = summ.get("max_abs_slope_bound")
    m.append(rf"\renewcommand{{\vSlopeBound}}{{{bound:.3f}}}" if bound else "")
    with open(os.path.join(gen, "v2.tex"), "w") as fh:
        fh.write("\n".join(m) + "\n")

    # held-out table
    body = []
    for mdl in sorted(held_m, key=lambda x: (pf.CLASS_ORDER.index(pf.FAMILY_CLASS[FAMILY[x]]),
                                             PARAMS[x])):
        cls = pf.CLASS_LABEL[pf.FAMILY_CLASS[FAMILY[mdl]]]
        def f2(x):
            return f"${x:.2f}$" if x < 0 else f"{x:.2f}"
        cells = " & ".join(f"{f2(pred[mdl][w])} & {f2(obs[mdl][w])}" for w in WIN)
        body.append(f"{pf._short(mdl)} & {cls} & {PARAMS[mdl]:.0f} & {cells} \\\\")
    with open(os.path.join(gen, "tab_heldout.tex"), "w") as fh:
        fh.write("\n".join(body) + "\n")
    with open(os.path.join(results_dir, "exploratory.json"), "w") as fh:
        json.dump(explo, fh, indent=1)
    return {"law_test": lt, "exploratory": explo}
