#!/usr/bin/env python3
"""HRH command line.

  python run_hrh.py brain    --subjects sub-01,...,sub-10   # EEG -> RDMs + ceilings (CPU)
  python run_hrh.py features --families all                # nets -> Gram matrices (GPU)
  python run_hrh.py features --families mae,ijepa,vjepa2   # objective-test models only
  python run_hrh.py eval                                   # full protocol -> results/ + paper/
  python run_hrh.py figures                                # synthetic validation + theory figs
  python run_hrh.py proxies                                # brain-free proxies for every model
  python run_hrh.py law                                    # fit + freeze the law (development models)
  python run_hrh.py predict                                # held-out predictions + hash to register
  python run_hrh.py analyze  --pool gap                    # one pooling only (exploration)
"""
from __future__ import annotations

import argparse
import os

ALL_SUBJECTS = ",".join(f"sub-{i:02d}" for i in range(1, 11))


def stage_brain(a):
    from hrh import data as pr
    from hrh import brain
    for s in a.subjects.split(","):
        path = pr.download_subject(s, a.eeg_dir)
        print("[brain]", brain.build_subject(s, path, os.path.join(a.out, "brain"),
                                             smooth=a.smooth))


def stage_features(a):
    from hrh import data as pr
    from hrh.features import ALL_LADDERS as LADDERS, extract_grams
    pr.ensure_images(a.img_dir, pr.CONFIG["N_TEST_IMAGES"])
    paths = pr.load_test_image_paths(a.img_dir, pr.CONFIG["N_TEST_IMAGES"])
    fams = LADDERS if a.families == "all" else {f: LADDERS[f] for f in a.families.split(",")}
    if a.models:
        want = set(a.models.split(","))
        fams = {f: [(m, p) for m, p in ms if m in want] for f, ms in LADDERS.items()}
    failed = []
    for fam, models in fams.items():
        for m, _p in models:
            try:
                print(f"[features] {fam}/{m}", flush=True)
                extract_grams(m, paths, a.device, os.path.join(a.out, "grams"))
            except Exception as exc:                 # fail soft per model, report at the end
                print(f"[skip] {m}: {exc!r}", flush=True)
                failed.append(m)
                import gc
                gc.collect()
                try:                                 # release GPU memory held by the failed model
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            if a.clean_cache:                        # free disk: weights are not needed again
                import glob
                import shutil
                hub = os.path.expanduser("~/.cache/huggingface/hub")
                tag = m[3:].replace("/", "--") if m.startswith("hf:") else f"timm--{m}"
                for d in glob.glob(os.path.join(hub, f"models--{tag}*")):
                    shutil.rmtree(d, ignore_errors=True)
    print(f"[features] done; {len(failed)} failed" + (": " + ", ".join(failed) if failed else ""))


def stage_analyze(a):
    from hrh import evaluate as ev
    res = ev.analyze(a.out, a.pool, n_depth=a.n_depth, n_boot=a.n_boot)
    ev.write_results(res, a.out)


def stage_eval(a):
    from hrh import evaluate as ev
    ev.evaluate(a.out, a.paper, n_boot=a.n_boot, n_depth=a.n_depth)


def stage_proxies(a):
    """Brain-free: Grams + images + human category labels only."""
    import glob
    import json

    import numpy as np

    from hrh import data as pr
    from hrh import proxies as px
    from hrh.features import PARAMS, model_rdms
    pr.ensure_images(a.img_dir, pr.CONFIG["N_TEST_IMAGES"])
    paths = pr.load_test_image_paths(a.img_dir, pr.CONFIG["N_TEST_IMAGES"])
    concepts = px.test_concepts(pr.CONFIG["IMAGE_METADATA_NPY"])
    cats = px.load_categories(px.ensure_categories(a.categories), concepts)
    low = px.features_to_rdm(px.gabor_features(paths))
    sem, valid = px.semantic_rdm(cats)
    out = {}
    for p in sorted(glob.glob(os.path.join(a.out, "grams", "*.npz"))):
        stem = os.path.basename(p)[:-4]
        name = next((m for m in PARAMS if m.replace(":", "_").replace("/", "_") == stem), None)
        if name is None:
            continue
        z = np.load(p, allow_pickle=False)
        ok = z[f"ok_{a.pool}"]
        if not ok.any():
            continue
        rdms, _n, _p = model_rdms(p, a.pool)
        layer = px.layer_proxies(z[f"gram_{a.pool}"][ok].astype(np.float64), rdms, low, sem, valid)
        out[name] = px.summarize(layer)
        print(f"[proxies] {name:<44} alpha {out[name]['alpha_min']:.2f}..{out[name]['alpha_max']:.2f}"
              f"  L {out[name]['L_max']:.3f}  S {out[name]['S_max']:.3f}", flush=True)
    with open(os.path.join(a.out, f"proxies_{a.pool}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def stage_law(a):
    """Fit the law on the development models and freeze it."""
    import csv
    import json

    from hrh import law as lw
    from hrh.features import DEVELOPMENT, FAMILY, PARAMS
    with open(os.path.join(a.out, f"proxies_{a.pool}.json")) as fh:
        prox = {m: v for m, v in json.load(fh).items() if m in DEVELOPMENT}
    with open(os.path.join(a.out, f"models_{a.pool}.csv")) as fh:
        scores = {r["model"]: {w: float(r[f"{w}_norm"]) for w in lw.WINDOWS}
                  for r in csv.DictReader(fh) if r["model"] in DEVELOPMENT}
    law = lw.fit(prox, scores, PARAMS, FAMILY)
    with open(os.path.join(a.out, "law.json"), "w") as fh:
        json.dump(law, fh, indent=1)
    for w, v in law["windows"].items():
        cv = v["cv"]
        print(f"[law] {w:<5} b={v['b']:.2f}  cv spearman: law {cv['law']['spearman']:+.2f}  "
              f"log-params {cv['log_params']['spearman']:+.2f}  "
              f"(r2 law {cv['law']['r2']:+.2f})", flush=True)


def stage_predict(a):
    """Predictions for the held-out models from proxies only; writes the file to register."""
    import json

    from hrh import law as lw
    from hrh.features import HELDOUT, PARAMS
    with open(os.path.join(a.out, "law.json")) as fh:
        law = json.load(fh)
    with open(os.path.join(a.out, f"proxies_{a.pool}.json")) as fh:
        prox = {m: v for m, v in json.load(fh).items() if m in HELDOUT}
    missing = sorted(HELDOUT - set(prox))
    if missing:
        print(f"[predict] WARNING: no proxies for {missing}")
    reg = {"law": law, "proxies": prox, "predictions": lw.predict(law, prox, PARAMS),
           "pool": a.pool}
    os.makedirs(a.reg_dir, exist_ok=True)
    with open(os.path.join(a.reg_dir, "predictions_heldout.json"), "w") as fh:
        json.dump(reg, fh, indent=1)
    digest = lw.sha256(reg)
    with open(os.path.join(a.reg_dir, "predictions_heldout.sha256"), "w") as fh:
        fh.write(digest + "  predictions_heldout.json\n")
    print(f"[predict] {len(prox)} held-out models -> {a.reg_dir}/predictions_heldout.json")
    print(f"[predict] sha256 {digest}")


def stage_figures(a):
    from hrh import evaluate as ev
    ev.write_synthetic_assets(a.paper)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = {n: sub.add_parser(n) for n in ("brain", "features", "analyze", "eval", "figures",
                                         "proxies", "law", "predict")}
    for n in ("brain", "features", "analyze", "eval", "proxies", "law", "predict"):
        p[n].add_argument("--out", default="results")
    for n in ("proxies", "law", "predict"):
        p[n].add_argument("--pool", default="grid4", choices=("grid4", "gap", "cls"))
    p["proxies"].add_argument("--img-dir", default="data/things/test_images")
    p["proxies"].add_argument("--categories", default="data/things/category53_wide-format.tsv")
    p["predict"].add_argument("--reg-dir", default="registration")
    for n in ("eval", "figures"):
        p[n].add_argument("--paper", default="paper")
    for n in ("analyze", "eval"):
        p[n].add_argument("--n-depth", type=int, default=8)
        p[n].add_argument("--n-boot", type=int, default=1000)
    p["brain"].add_argument("--subjects", default=ALL_SUBJECTS)
    p["brain"].add_argument("--eeg-dir", default="data/eeg")
    p["brain"].add_argument("--smooth", type=int, default=1, help="boxcar width (samples)")
    p["features"].add_argument("--families", default="all")
    p["features"].add_argument("--img-dir", default="data/things/test_images")
    p["features"].add_argument("--device", default="cuda")
    p["features"].add_argument("--models", default=None, help="comma list (overrides families)")
    p["features"].add_argument("--clean-cache", action="store_true",
                               help="delete each model's downloaded weights after extraction")
    p["analyze"].add_argument("--pool", default="grid4", choices=("grid4", "gap", "cls"))
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
