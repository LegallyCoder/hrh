"""Evaluation pipeline: brain RDMs + model Grams -> metrics, tests, paper assets.

`analyze()` computes everything for one pooling choice; `evaluate()` runs the
full pre-registered protocol (all poolings, all metrics) and writes
  results/            CSV / JSON / NPZ for every analysis
  paper/generated/    LaTeX macros and tables  (\\input by the paper)
  paper/figures/      vector figures           (\\includegraphics by the paper)
The paper never contains a hand-typed number from the real-data analysis.
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

from . import brain, plotting, trajectory as tr
from . import law as lw
from .features import (CONTROLLED, FAMILY, HELDOUT, OBJECTIVE_LADDERS, PARAMS, PRIMARY,
                       REFERENCE_BY_SIZE, SIZE_CLASS, model_rdms)

REG_DIR = os.environ.get("HRH_REG_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registration"))


def registered_predictions(reg_dir: str | None = None) -> dict | None:
    """The held-out predictions, if registered: the JSON must match its committed hash.
    Until then held-out models are excluded from every brain analysis."""
    reg_dir = reg_dir or REG_DIR
    pj = os.path.join(reg_dir, "predictions_heldout.json")
    ph = os.path.join(reg_dir, "predictions_heldout.sha256")
    if not (os.path.exists(pj) and os.path.exists(ph)):
        return None
    with open(pj) as fh:
        reg = json.load(fh)
    with open(ph) as fh:
        want = fh.read().split()[0]
    if lw.sha256(reg) != want:
        raise SystemExit("registration/predictions_heldout.json does not match its hash")
    return reg

POOLS = ("grid4", "gap", "cls")          # grid4 is the pre-registered primary
METRICS = ("rsa", "cka", "mknn")         # rsa is the pre-registered primary


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_subjects(results_dir: str) -> list:
    subs = [brain.SubjectRDMs.load(p)
            for p in sorted(glob.glob(os.path.join(results_dir, "brain", "brain_*.npz")))]
    if not subs:
        raise SystemExit(f"no {results_dir}/brain/brain_*.npz — run the `brain` stage first")
    return subs


def load_models(results_dir: str, pool: str) -> dict[str, np.ndarray]:
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "grams", "*.npz"))):
        stem = os.path.basename(p)[:-4]
        name = next((m for m in PARAMS if m.replace(":", "_").replace("/", "_") == stem), None)
        if name is None:
            continue
        if name in HELDOUT and registered_predictions() is None:
            continue                     # no brain analysis before predictions are registered
        rd, _names, _params = model_rdms(p, pool)
        if len(rd):
            out[name] = rd
    if not out:
        raise SystemExit(f"no usable {results_dir}/grams/*.npz for pool={pool}")
    return out


def _summ(boot: np.ndarray) -> dict:
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return {"mean": None, "ci95": None, "p_gt0": None}
    return {"mean": float(boot.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_gt0": float((boot > 0).mean())}


# ---------------------------------------------------------------------------
# One analysis (one pooling)
# ---------------------------------------------------------------------------

def analyze(results_dir: str, pool: str = "grid4", n_depth: int = 8, n_boot: int = 1000,
            subjects=None, metrics=METRICS, verbose: bool = True) -> dict:
    subjects = subjects or load_subjects(results_dir)
    models = load_models(results_dir, pool)
    times = subjects[0].times_ms
    masks = {s.subject: tr.reliable_mask(times, s.split_half_r) for s in subjects}

    rows, mats, curves, selected = [], {}, {}, {s.subject: {} for s in subjects}
    robust = {m: {} for m in metrics if m != "rsa"}       # metric -> model -> subj -> win -> score
    for name, rd in models.items():
        per = []
        for s in subjects:
            A = tr.alignment_matrix(rd, s.full, "rsa")
            An = tr.normalize(A, s.ceiling, masks[s.subject])
            Ad = tr.subsample_depth(An, n_depth)
            ws = tr.window_scores(rd, s)
            selected[s.subject][name] = {w: (v["layer_a"], v["layer_b"]) for w, v in ws.items()}
            per.append(dict(A=A, curve=np.nanmax(np.where(np.isfinite(An), An, -np.inf), 0),
                            cov=tr.coverage_auc(Ad), covc=tr.coverage_curve(Ad),
                            hier=tr.hierarchy_slope(A, times, masks[s.subject]),
                            dtw=tr.dtw_distance(Ad), ws=ws,
                            pre=float(np.abs(A[:, times < 0]).mean())))
            for m in robust:
                robust[m].setdefault(name, {})[s.subject] = {
                    w: v["score"] for w, v in tr.window_scores(rd, s, metric=m).items()}
        mats[name] = np.mean([q["A"] for q in per], 0)
        cv = np.stack([q["curve"] for q in per])
        cv = np.where(np.isfinite(cv), cv, np.nan)
        with np.errstate(all="ignore"):
            ok = np.isfinite(cv).any(0)
            curves[name] = np.full(cv.shape[1], np.nan)
            curves[name][ok] = np.nanmean(cv[:, ok], 0)
        row = {"model": name, "family": FAMILY[name], "params_m": PARAMS[name],
               "n_layers": len(rd),
               "coverage_auc": float(np.mean([q["cov"] for q in per])),
               "hierarchy": float(np.nanmean([q["hier"] for q in per])),
               "D_HRH": float(np.mean([q["dtw"] for q in per])),
               "pre_onset_abs_rsa": float(np.mean([q["pre"] for q in per]))}
        for w in tr.WINDOWS:
            row[f"{w}_norm"] = float(np.mean([q["ws"][w]["norm"] for q in per]))
            row[f"{w}_raw"] = float(np.mean([q["ws"][w]["score"] for q in per]))
            row[f"{w}_depth"] = float(np.mean([q["ws"][w]["layer_b"] / max(len(rd) - 1, 1)
                                               for q in per]))
        row["coverage_curve"] = np.mean([q["covc"] for q in per], 0).round(3).tolist()
        rows.append(row)
        if verbose:
            print(f"[{pool}] {name:<44} cov {row['coverage_auc']:.3f}  D {row['D_HRH']:.3f}  "
                  + "  ".join(f"{w} {row[f'{w}_norm']:.3f}" for w in tr.WINDOWS), flush=True)

    prim = [r for r in rows if r["model"] in PRIMARY]          # pre-registered ladders only
    fam = {r["model"]: r["family"] for r in prim}
    par = {r["model"]: r["params_m"] for r in prim}

    def only(sc, keep):
        return {m: v for m, v in sc.items() if m in keep}
    obj_members = {f: [m for m, _ in ms if m in models] for f, ms in OBJECTIVE_LADDERS.items()}
    obj_all = [m for ms in obj_members.values() for m in ms]
    par_all = {r["model"]: r["params_m"] for r in rows}
    fam_all = {r["model"]: r["family"] for r in rows}
    stats = {"slope": lambda sc: tr.family_slope(only(sc, fam), par, fam)}
    if obj_all:
        stats["objective"] = lambda sc: tr.objective_contrast(sc, obj_all, SIZE_CLASS,
                                                              REFERENCE_BY_SIZE)
        for f, ms in obj_members.items():
            if ms:
                stats[f"objective_{f}"] = (lambda sc, ms=ms: tr.objective_contrast(
                    sc, ms, SIZE_CLASS, REFERENCE_BY_SIZE))
        stats["slope_new_ladders"] = lambda sc: tr.family_slope(
            only(sc, set(obj_all)), par_all, fam_all)
    reg = registered_predictions() if pool == "grid4" else None
    held = [m for m in models if m in HELDOUT]
    if reg is not None and len(held) >= 3:
        pred = reg["predictions"]
        members = [m for m in held if m in pred]
        mrng = np.random.default_rng(1)

        def draw():
            """Models are a sample too: resample them in every bootstrap draw, so the
            interval reflects how few held-out models there are."""
            return [members[i] for i in mrng.integers(0, len(members), len(members))]

        def rho(sc, w, ms, key=""):
            return lw._spearman([pred[m][w + key] for m in ms], [sc[m] for m in ms])

        def f_dif(sc, w):
            ms = draw()
            return rho(sc, w, ms) - rho(sc, w, ms, "_params_baseline")
        f_law = lambda sc, w: rho(sc, w, draw())                          # noqa: E731
        f_par = lambda sc, w: rho(sc, w, draw(), "_params_baseline")      # noqa: E731
        for k, f in (("law_rho", f_law), ("params_rho", f_par), ("law_minus_params", f_dif)):
            f.window_aware = True
            stats[k] = f
        stats["slope_heldout"] = lambda sc: tr.family_slope(
            only(sc, set(held)), par_all, fam_all)
    ctl = {o: sorted(m for m in models if m in CONTROLLED and FAMILY[m] == f"ctl-{o}")
           for o in ("sup", "mae", "jepa", "data2vec")}
    has_ctl = all(ctl.values())
    if has_ctl:
        def mean_of(sc, ms):
            return float(np.mean([sc[m] for m in ms]))
        latent = ctl["jepa"] + ctl["data2vec"]
        stats["ctl_latent_minus_sup"] = lambda sc: mean_of(sc, latent) - mean_of(sc, ctl["sup"])
        stats["ctl_latent_minus_mae"] = lambda sc: mean_of(sc, latent) - mean_of(sc, ctl["mae"])
        for o, ms in ctl.items():
            stats[f"ctl_mean_{o}"] = (lambda sc, ms=ms: mean_of(sc, ms))
    boots = tr.bootstrap_window_stats(models, subjects, selected, stats, n_boot=n_boot)
    slopes = {w: _summ(b) for w, b in boots["slope"].items()}
    summary = {
        "pool": pool, "n_subjects": len(subjects), "n_models": len(prim),
        "n_families_with_ladder": sum(1 for f in set(fam.values())
                                      if sum(v == f for v in fam.values()) >= 2),
        "slope_vs_log10_params": slopes,
        "max_abs_slope_bound": max(max(abs(x) for x in v["ci95"]) for v in slopes.values()
                                   if v["ci95"]) if all(v["ci95"] for v in slopes.values())
        else None,
        **{f"{k}_slope": tr.family_slope({r["model"]: r[k] for r in prim}, par, fam)
           for k in ("coverage_auc", "D_HRH", "hierarchy")},
        "max_pre_onset_abs_rsa": max(r["pre_onset_abs_rsa"] for r in rows),
        "reliable_bins_ms": {s.subject: ([float(times[masks[s.subject]].min()),
                                          float(times[masks[s.subject]].max())]
                                         if masks[s.subject].any() else None)
                             for s in subjects},
    }
    if obj_all:
        ot = {k: {w: _summ(b) for w, b in boots[k].items()}
              for k in boots if k.startswith("objective")}
        summary["objective_test"] = {
            "members": obj_members, "n_models": len(obj_all),
            "contrast": ot["objective"],
            "by_family": {k[len("objective_"):]: v for k, v in ot.items() if k != "objective"},
            "verdict": objective_verdict(ot["objective"]),
            # registered after the I-JEPA and MAE data were seen, before V-JEPA 2 was run:
            # V-JEPA 2 replicates I-JEPA (same rule as the pooled test, on its own contrast)
            "vjepa2_replication": (objective_verdict(ot["objective_vjepa2"])
                                   if "objective_vjepa2" in ot else None),
            "slope_new_ladders": {w: _summ(b) for w, b in boots["slope_new_ladders"].items()},
        }
    if "law_rho" in boots:
        lt = {k: {w: _summ(b) for w, b in boots[k].items()}
              for k in ("law_rho", "params_rho", "law_minus_params", "slope_heldout")}
        summary["law_test"] = {"n_models": len(held), "members": held,
                               "prediction_sha256": lw.sha256(reg), **lt,
                               "verdict": lw.law_verdict(lt["law_rho"], lt["law_minus_params"])}
    if has_ctl:
        cs = {k: {w: _summ(b) for w, b in boots[k].items()} for k in boots if k.startswith("ctl_")}
        summary["controlled"] = {"members": ctl, **cs,
                                 "verdict": controlled_verdict(cs["ctl_latent_minus_sup"],
                                                               cs["ctl_latent_minus_mae"])}
    # metric robustness: slope per subject; agreement of sign with the RSA slope
    summary["robustness"] = {}
    for m, by_model in robust.items():
        summary["robustness"][m] = {}
        for w in tr.WINDOWS:
            per_subj = [tr.family_slope({mod: by_model[mod][s.subject][w] for mod in by_model
                                         if mod in fam}, par, fam) for s in subjects]
            mean = float(np.nanmean(per_subj))
            ref = slopes[w]["mean"]
            agree = (float(np.nanmean([np.sign(x) == np.sign(ref) for x in per_subj]))
                     if ref is not None else None)
            summary["robustness"][m][w] = {"mean_slope": mean, "subject_sign_agreement": agree}
    summary["verdict"] = verdict(slopes)
    return dict(rows=rows, mats=mats, curves=curves, summary=summary, subjects=subjects,
                masks=masks,
                times=times, slopes=slopes)


def controlled_verdict(vs_sup: dict, vs_mae: dict) -> dict:
    """Pre-specified rule for the controlled objective experiment (hrh/train.py).
    H1 (primary): latent prediction (JEPA and data2vec, seeds averaged) exceeds BOTH
    supervised and MAE networks in the early window (lower 95% bounds > 0).
    H2: the same in the late window."""
    def up(d, w):
        return bool((d[w]["ci95"] or [np.nan])[0] > 0)
    h1 = up(vs_sup, "early") and up(vs_mae, "early")
    h2 = up(vs_sup, "late") and up(vs_mae, "late")
    return {"H1_early": h1, "H2_late": h2,
            "label": "supported" if h1 else "not supported"}


def objective_verdict(contrast: dict) -> dict:
    """Pre-registered rule for the objective test (Proposition 3): dense / predictive
    objectives beat size-matched single-readout models in BOTH off-bottleneck
    windows (lower 95% bound > 0 for early and late). No prediction for 'mid'."""
    lo = {w: (contrast[w]["ci95"] or [np.nan, np.nan])[0] for w in ("early", "late")}
    hi = {w: (contrast[w]["ci95"] or [np.nan, np.nan])[1] for w in ("early", "late")}
    n_up = sum(lo[w] > 0 for w in lo)
    label = ("supported" if n_up == 2 else "partial" if n_up == 1
             else "reversed" if all(hi[w] < 0 for w in hi) else "not supported")
    return {"label": label, "early_up": bool(lo["early"] > 0), "late_up": bool(lo["late"] > 0)}


def verdict(slopes: dict) -> dict:
    def lo(w):
        return (slopes[w]["ci95"] or [np.nan, np.nan])[0]

    def hi(w):
        return (slopes[w]["ci95"] or [np.nan, np.nan])[1]
    prh = all(lo(w) > 0 for w in slopes)
    hrh = lo("mid") > 0 and hi("early") <= 0 and hi("late") <= 0
    hrh_weak = lo("mid") > 0 and (hi("early") <= 0 or hi("late") <= 0)
    label = ("HRH" if hrh else "PRH" if prh else "HRH-partial" if hrh_weak else "inconclusive")
    return {"PRH_all_windows_positive": bool(prh), "HRH_strict": bool(hrh),
            "HRH_partial": bool(hrh_weak), "label": label}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_results(res: dict, results_dir: str) -> None:
    pool = res["summary"]["pool"]
    rows = res["rows"]
    with open(os.path.join(results_dir, f"models_{pool}.csv"), "w", newline="") as f:
        cols = [k for k in rows[0] if k != "coverage_curve"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(results_dir, f"summary_{pool}.json"), "w") as f:
        json.dump(res["summary"], f, indent=2)
    np.savez_compressed(os.path.join(results_dir, f"A_matrices_{pool}.npz"),
                        times_ms=res["times"],
                        **{k.replace(":", "_"): v for k, v in res["mats"].items()})


def _fmt_ci(s: dict) -> str:
    if not s or s.get("mean") is None:
        return r"\tbd"
    return f"${s['mean']:+.3f}$ [${s['ci95'][0]:+.3f}$, ${s['ci95'][1]:+.3f}$]"


def _num(x, nd=3, sign=False) -> str:
    if x is None or not np.isfinite(x):
        return r"\tbd"
    return f"${x:+.{nd}f}$" if sign else f"{x:.{nd}f}"


def _texname(m: str) -> str:
    return plotting.short_name(m).replace("_", r"\_")


def finding_sentence(summary: dict) -> str:
    """Abstract sentence chosen by the verdict — written for every outcome,
    including the one that falsifies HRH."""
    s = summary["slope_vs_log10_params"]
    e, m, lt = (_fmt_ci(s[w]) for w in ("early", "mid", "late"))
    head = (f"Across {summary['n_subjects']} subjects and {summary['n_models']} models in "
            f"{summary['n_families_with_ladder']} scaling ladders, ")
    lab = summary["verdict"]["label"]
    if lab == "HRH":
        return (head + f"scaling increased alignment in the semantic window ($\\beta$={m}) "
                f"but not in the early ($\\beta$={e}) or late ($\\beta$={lt}) windows, "
                "as HRH predicts and a static reading of PRH does not.")
    if lab == "PRH":
        return (head + f"scaling increased alignment in every window (early {e}, semantic {m}, "
                f"late {lt}), contradicting the HRH prediction.")
    if lab == "HRH-partial":
        return (head + f"scaling increased semantic-window alignment ($\\beta$={m}) while "
                f"early ({e}) and late ({lt}) windows did not uniformly follow, "
                "partially supporting HRH.")
    bound = summary.get("max_abs_slope_bound")
    tail = (f"; every 95\\% interval excludes changes larger than {bound:.3f} per decade"
            if bound is not None else "")
    return (head + "alignment did not change reliably with scale in any window (slope per "
            f"decade of parameters: early {e}, semantic {m}, late {lt}{tail}).")


def objective_sentence(test: dict) -> str:
    c = test["contrast"]
    e, m, lt = (_fmt_ci(c[w]) for w in ("early", "mid", "late"))
    lab = test["verdict"]["label"]
    head = (f"Against size-matched single-readout networks, {test['n_models']} models trained "
            "with dense predictive objectives (MAE, I-JEPA, V-JEPA~2) ")
    if lab == "supported":
        return head + (f"aligned better in both off-bottleneck windows (early $\\Delta$={e}, "
                       f"late $\\Delta$={lt}; semantic {m}), as Proposition~3 predicts.")
    if lab == "partial":
        return head + (f"aligned better in one off-bottleneck window but not the other "
                       f"(early $\\Delta$={e}, late $\\Delta$={lt}; semantic {m}).")
    if lab == "reversed":
        return head + (f"aligned worse in both off-bottleneck windows (early $\\Delta$={e}, "
                       f"late $\\Delta$={lt}; semantic {m}), contrary to Proposition~3.")
    return head + (f"did not align reliably better outside the bottleneck (early $\\Delta$={e}, "
                   f"late $\\Delta$={lt}; semantic {m}), so Proposition~3's prediction "
                   "was not confirmed.")


def write_paper_assets(primary: dict, others: list[dict], paper_dir: str) -> None:
    gen = os.path.join(paper_dir, "generated")
    figs = os.path.join(paper_dir, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    s = primary["summary"]
    sl = s["slope_vs_log10_params"]
    prows = [r for r in primary["rows"] if r["model"] in PRIMARY]
    pmats = {k: v for k, v in primary["mats"].items() if k in PRIMARY}
    lines = [r"% AUTO-GENERATED by `python run_hrh.py eval` — do not edit by hand.",
             r"\hrhresultstrue",
             rf"\renewcommand{{\hrhNsubj}}{{{s['n_subjects']}}}",
             rf"\renewcommand{{\hrhNmodels}}{{{s['n_models']}}}",
             rf"\renewcommand{{\hrhNladders}}{{{s['n_families_with_ladder']}}}",
             rf"\renewcommand{{\hrhSlopeEarly}}{{{_fmt_ci(sl['early'])}}}",
             rf"\renewcommand{{\hrhSlopeMid}}{{{_fmt_ci(sl['mid'])}}}",
             rf"\renewcommand{{\hrhSlopeLate}}{{{_fmt_ci(sl['late'])}}}",
             rf"\renewcommand{{\hrhCovSlope}}{{{_num(s['coverage_auc_slope'], sign=True)}}}",
             rf"\renewcommand{{\hrhDSlope}}{{{_num(s['D_HRH_slope'], sign=True)}}}",
             rf"\renewcommand{{\hrhHierSlope}}{{{_num(s['hierarchy_slope'], sign=True)}}}",
             rf"\renewcommand{{\hrhMaxPre}}{{{_num(s['max_pre_onset_abs_rsa'])}}}",
             rf"\renewcommand{{\hrhVerdict}}{{{s['verdict']['label']}}}",
             rf"\renewcommand{{\hrhFinding}}{{{finding_sentence(s)}}}",
             rf"\renewcommand{{\hrhSlopeBound}}{{{_num(s.get('max_abs_slope_bound'))}}}"]
    ot = s.get("objective_test")
    if ot:
        c, nl = ot["contrast"], ot["slope_new_ladders"]
        lines += [r"\hrhobjtrue",
                  rf"\renewcommand{{\hrhObjN}}{{{ot['n_models']}}}",
                  rf"\renewcommand{{\hrhObjEarly}}{{{_fmt_ci(c['early'])}}}",
                  rf"\renewcommand{{\hrhObjMid}}{{{_fmt_ci(c['mid'])}}}",
                  rf"\renewcommand{{\hrhObjLate}}{{{_fmt_ci(c['late'])}}}",
                  rf"\renewcommand{{\hrhObjVerdict}}{{{ot['verdict']['label']}}}",
                  rf"\renewcommand{{\hrhObjFinding}}{{{objective_sentence(ot)}}}",
                  rf"\renewcommand{{\hrhNewSlopeEarly}}{{{_fmt_ci(nl['early'])}}}",
                  rf"\renewcommand{{\hrhNewSlopeMid}}{{{_fmt_ci(nl['mid'])}}}",
                  rf"\renewcommand{{\hrhNewSlopeLate}}{{{_fmt_ci(nl['late'])}}}"]
        tab = []
        for key, lab in [("all", "All dense objectives")] + [
                (f, plotting.OBJ_LABEL[f]) for f in plotting.OBJ_ORDER if f in ot["by_family"]]:
            d = ot["contrast"] if key == "all" else ot["by_family"][key]
            n = ot["n_models"] if key == "all" else len(ot["members"][key])
            tab.append(f"{lab} & {n} & " + " & ".join(_fmt_ci(d[w]) for w in tr.WINDOWS)
                       + r" \\")
            if key == "all":
                tab.append(r"\addlinespace[2pt]")
        open(os.path.join(gen, "tab_objective.tex"), "w").write("\n".join(tab) + "\n")
        plotting.fig_objective(primary["rows"], ot, tr.WINDOWS, SIZE_CLASS, REFERENCE_BY_SIZE,
                               os.path.join(figs, "objective.pdf"))
    open(os.path.join(gen, "results.tex"), "w").write("\n".join(lines) + "\n")

    # per-model table (primary pooling)
    body = []
    for f in plotting.FAMILY_ORDER:
        rr = sorted([r for r in prows if r["family"] == f], key=lambda r: r["params_m"])
        for i, r in enumerate(rr):
            fam = plotting.FAMILY_LABEL[f] if i == 0 else ""
            body.append(f"{fam} & {_texname(r['model'])} & {r['params_m']:.0f} & "
                        f"{r['early_norm']:.3f} & {r['mid_norm']:.3f} & {r['late_norm']:.3f} & "
                        f"{r['coverage_auc']:.3f} & {r['D_HRH']:.3f} \\\\")
        if rr:
            body.append(r"\addlinespace[2pt]")
    open(os.path.join(gen, "tab_models.tex"), "w").write("\n".join(body) + "\n")

    # robustness table: pooling x metric -> slope signs per window
    rob = []
    for res in [primary] + others:
        ss = res["summary"]
        cells = " & ".join(_fmt_ci(ss["slope_vs_log10_params"][w]) for w in tr.WINDOWS)
        rob.append(f"{ss['pool']} & RSA & {cells} \\\\")
        for m, d in ss["robustness"].items():
            cells = " & ".join(
                f"{_num(d[w]['mean_slope'], sign=True)} "
                f"({_num(d[w]['subject_sign_agreement'], 2)})" for w in tr.WINDOWS)
            rob.append(f" & {m.upper() if m == 'cka' else 'mkNN'} & {cells} \\\\")
        rob.append(r"\addlinespace[2pt]")
    open(os.path.join(gen, "tab_robust.tex"), "w").write("\n".join(rob) + "\n")

    common = np.all(np.stack([primary["masks"][x.subject] for x in primary["subjects"]]), 0)
    plotting.fig_windows(prows, primary["curves"], primary["times"], tr.WINDOWS,
                         os.path.join(figs, "windows.pdf"))
    plotting.fig_scaling(prows, tr.WINDOWS, os.path.join(figs, "scaling.pdf"), sl, s)
    plotting.fig_layer_time(pmats, primary["times"], PARAMS, FAMILY,
                            os.path.join(figs, "layer_time.pdf"), common)
    plotting.fig_ceiling(primary["subjects"], os.path.join(figs, "ceiling.pdf"),
                         primary["masks"], tr.WINDOWS)


def write_synthetic_assets(paper_dir: str) -> None:
    """Validation + theory figures and their numbers (both from synthetic ground truth)."""
    gen = os.path.join(paper_dir, "generated")
    figs = os.path.join(paper_dir, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    v = plotting.fig_validation(os.path.join(figs, "validation.pdf"))
    t = plotting.fig_theory(os.path.join(figs, "theory.pdf"))
    m = {"synCovHier": v["hier"]["cov"], "synCovStatic": v["static"]["cov"],
         "synCovNoise": v["noise"]["cov"], "synDHier": v["hier"]["dtw"],
         "synDStatic": v["static"]["dtw"], "synMidHier": v["hier"]["mid"],
         "synMidStatic": v["static"]["mid"], "thCeiling": t["ceiling"],
         "thEarlyInit": t["early_init"], "thEarlySingle": t["early_single"],
         "thEarlyMulti": t["early_multi"]}
    lines = [r"% AUTO-GENERATED by `python run_hrh.py figures` — do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{val:.2f}}}" for k, val in m.items()]
    lines += [rf"\newcommand{{\thNuisance}}{{{t['nuisance_final']:.0e}}}",
              rf"\newcommand{{\thCollapse}}{{{t['collapse_final']:.0e}}}"]
    open(os.path.join(gen, "synthetic.tex"), "w").write("\n".join(lines) + "\n")


def evaluate(results_dir: str, paper_dir: str, n_boot: int = 1000, n_depth: int = 8,
             pools: tuple = POOLS) -> dict:
    subjects = load_subjects(results_dir)
    out = {}
    for pool in pools:
        try:
            out[pool] = analyze(results_dir, pool, n_depth=n_depth,
                                n_boot=n_boot if pool == POOLS[0] else max(n_boot // 4, 100),
                                subjects=subjects)
        except Exception as exc:                       # a secondary pooling must never
            if pool == POOLS[0]:                       # block the primary paper assets
                raise
            print(f"[eval] skip pool={pool}: {exc!r}")
            continue
        except SystemExit as exc:
            print(f"[eval] skip pool={pool}: {exc}")
            continue
        write_results(out[pool], results_dir)
    if POOLS[0] not in out:
        raise SystemExit("primary pooling failed")
    write_paper_assets(out[POOLS[0]], [out[p] for p in POOLS[1:] if p in out], paper_dir)
    v = out[POOLS[0]]["summary"]
    print(json.dumps({"slopes": v["slope_vs_log10_params"], "verdict": v["verdict"],
                      "objective_test": {k: v.get("objective_test", {}).get(k)
                                         for k in ("contrast", "verdict")}}, indent=2))
    return out
