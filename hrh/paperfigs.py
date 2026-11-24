"""Figures for the paper: theory, scale vs objective, the law and its held-out test.

Visual system (shared with plotting.py: serif text, recessive axes, 4 px marks)
------------------------------------------------------------------------------
* objective class -> hue (validated palette; all pairs pass normal-vision and
  CVD checks), supervised models are the neutral reference class
* every class also has its own marker, so identity never relies on colour
* development models are drawn hollow, held-out models filled
"""
from __future__ import annotations

import numpy as np

from .plotting import BAND, GRID, INK, INK2, MUTED, TEXT_W, WINDOW_LABEL, setup, tint, title, ygrid

CLASS_ORDER = ["latent", "pixel", "language", "selfdistill", "supervised"]
CLASS_COLOR = {"latent": "#2a78d6", "pixel": "#eda100", "language": "#e87ba4",
               "selfdistill": "#008300", "supervised": "#8a8984"}
CLASS_EDGE = {k: v for k, v in CLASS_COLOR.items()} | {"pixel": "#b57b00"}  # contrast relief
CLASS_MARKER = {"latent": "o", "pixel": "s", "language": "D", "selfdistill": "^",
                "supervised": "v"}
CLASS_LABEL = {"latent": "latent prediction", "pixel": "pixel / generative",
               "language": "language-supervised", "selfdistill": "self-distillation",
               "supervised": "supervised"}
FAMILY_CLASS = {
    "classic-cnn": "supervised", "resnet-in1k": "supervised", "convnext-in22k": "supervised",
    "vit-sup": "supervised", "dinov2": "selfdistill", "clip-laion": "language",
    "mae": "pixel", "ijepa": "latent", "vjepa2": "latent",
    "dino": "selfdistill", "fcmae": "pixel", "eva02-mim": "latent", "siglip": "language",
    "aimv2": "pixel", "data2vec": "latent", "beit-pt22k": "latent",
}
WIN = ("early", "mid", "late")


def _legend(fig, classes, y=-0.06, extra=()):
    """Two rows: objective classes, then encodings (hollow/filled, line keys)."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], lw=0, marker=CLASS_MARKER[c], ms=5, mfc=CLASS_COLOR[c],
                mec=CLASS_EDGE[c], mew=0.6) for c in classes]
    fig.legend(h, [CLASS_LABEL[c] for c in classes], loc="lower center", ncol=len(h),
               bbox_to_anchor=(0.5, y), columnspacing=1.4, handletextpad=0.35, frameon=False)
    if extra:
        fig.legend([e[0] for e in extra], [e[1] for e in extra], loc="lower center",
                   ncol=len(extra), bbox_to_anchor=(0.5, y - 0.075), columnspacing=1.4,
                   handletextpad=0.35, frameon=False, labelcolor=INK2)


def _halo(t):
    import matplotlib.patheffects as pe
    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    return t


def _mark(ax, x, y, cls, held, s=26, z=3):
    col, edge = CLASS_COLOR[cls], CLASS_EDGE[cls]
    ax.scatter(x, y, s=s, marker=CLASS_MARKER[cls], zorder=z, lw=0.9,
               facecolor=col if held else "white", edgecolor=edge if held else col)


# ---------------------------------------------------------------------------
# Figure: theory (spectral matching and spectral trajectories)
# ---------------------------------------------------------------------------

def fig_spectral(out_path: str, proxies: dict | None = None, family: dict | None = None,
                 b_windows: dict | None = None) -> dict:
    from . import theory as th
    plt, _ = setup()
    sig = th.make_spectrum(d=16, decay=1.0)
    lat = th.psd_power(sig, -0.5)
    depth = 8
    runs = {"pixel": th.train_aniso(sig, np.eye(16), depth=depth, lr=0.01),
            "latent": th.train_aniso(sig, lat, depth=depth, lr=0.01)}
    ex = {k: [th.spectral_exponent(g, 1, 12) for g in th.whitened_grams(w, sig)]
          for k, w in runs.items()}
    b_windows = b_windows or {"early": 0.85, "mid": 0.25, "late": 0.55}

    ncol = 3 if proxies else 2
    fig = plt.figure(figsize=(TEXT_W, 2.25))
    gs = fig.add_gridspec(1, ncol, width_ratios=[1, 1, 1.45][:ncol], wspace=0.42)

    # (a) spectral trajectories through depth
    ax = fig.add_subplot(gs[0])
    t = np.arange(1, depth + 1)
    ax.plot([0, depth], [1, 0], color=GRID, lw=5, solid_capstyle="round", zorder=0)
    ax.plot(t, ex["pixel"], color=CLASS_EDGE["pixel"], marker="s", ms=3.6, lw=1.6)
    ax.plot(t, ex["latent"], color=CLASS_COLOR["latent"], marker="o", ms=3.6, lw=1.6)
    ax.text(depth + 0.25, ex["pixel"][-1], "pixel\nreconstruction", fontsize=6.3,
            color=CLASS_EDGE["pixel"], va="center")
    ax.text(depth + 0.25, ex["latent"][-1] + 0.06, "whitened\nlatent target", fontsize=6.3,
            color=CLASS_COLOR["latent"], va="center")
    ax.text(5.0, 0.68, r"$1 - t/T$", fontsize=6.6, color=MUTED)
    ax.set(xlim=(0.4, depth + 3.2), ylim=(-0.1, 1.15), xticks=[1, 4, 8],
           xlabel="layer $t$ (of $T=8$)", ylabel=r"spectral exponent of layer")
    ax.spines["bottom"].set_bounds(1, depth)
    ygrid(ax)
    title(ax, "a", "The objective fixes the path")

    # (b) spectral matching: inverted U
    ax = fig.add_subplot(gs[1])
    eigs = np.arange(1, 201, dtype=float) ** -1.0
    a = np.linspace(-0.2, 1.6, 181)
    lift = {"mid": 1.03, "late": 1.075, "early": 1.03}
    for w in sorted(WIN, key=lambda w: b_windows[w]):
        b = b_windows[w]
        c = [th.spectral_cka(eigs, x, b) for x in a]
        shade = {"early": 1.0, "mid": 0.4, "late": 0.68}[w]
        ax.plot(a, c, color=tint(INK, shade), lw=1.5)
        ax.plot([b, b], [1.0, lift[w] - 0.012], color=tint(INK, shade), lw=0.6)
        ax.scatter([b], [1.0], s=14, color=tint(INK, shade), zorder=3)
        ax.text(b, lift[w], WINDOW_LABEL[w], ha="center", fontsize=6.1,
                color=INK2, style="italic")
    ax.set(xlim=(-0.2, 1.6), ylim=(0.55, 1.12), xlabel="model exponent $a$",
           ylabel="CKA with stage of exponent $b$")
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ygrid(ax)
    title(ax, "b", "Alignment peaks at a match")

    # (c) measured exponent ranges per model vs the windows' fitted exponents
    if proxies:
        ax = fig.add_subplot(gs[2])
        fam = family or {}
        ms = sorted(proxies, key=lambda m: (CLASS_ORDER.index(FAMILY_CLASS.get(fam.get(m), "supervised")),
                                            np.median(proxies[m]["alpha"])))
        xs, prev, x = [], None, 0.0
        for m in ms:                                   # a gap between objective classes
            cls = FAMILY_CLASS.get(fam.get(m), "supervised")
            x += 1.0 + (1.6 if prev is not None and cls != prev else 0.0)
            xs.append(x)
            prev = cls
        xmax = xs[-1] + 1
        for w in WIN:
            ax.axhline(b_windows[w], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
            ax.text(xmax + 0.8, b_windows[w] + 0.02, WINDOW_LABEL[w], fontsize=6.1, color=INK2,
                    va="bottom", style="italic")
        top = 2.5
        for x, m in zip(xs, ms):
            cls = FAMILY_CLASS.get(fam.get(m), "supervised")
            al = np.asarray(proxies[m]["alpha"], float)
            al = al[np.isfinite(al)]
            ax.plot([x, x], [al.min(), min(al.max(), top)], color=tint(CLASS_COLOR[cls], 0.6),
                    lw=1.7, solid_capstyle="round", zorder=2)
            if al.max() > top:
                ax.text(x, top + 0.04, "↑", ha="center", va="bottom", fontsize=5.5,
                        color=CLASS_EDGE[cls])
            ax.scatter(x, al[-1], s=9, color=CLASS_EDGE[cls], zorder=3, lw=0)
        ax.set(xlim=(0, xmax + 4.2), ylim=(0, top + 0.2), xticks=[],
               xlabel=f"{len(ms)} models, grouped by objective",
               ylabel="measured layer exponents")
        ax.spines["bottom"].set_visible(False)
        ygrid(ax)
        title(ax, "c", "Measured exponents of real networks")
        _legend(fig, CLASS_ORDER, y=-0.14)
    fig.savefig(out_path)
    plt.close(fig)
    return {"pixel": ex["pixel"], "latent": ex["latent"]}


# ---------------------------------------------------------------------------
# Figure: scale versus objective
# ---------------------------------------------------------------------------

def fig_scale_objective(rows: list[dict], slopes: dict, out_path: str,
                        heldout: set | None = None, windows: dict | None = None):
    """(a) within-family scaling slopes with 95% CIs; (b-d) window score against
    size for every model, ladders joined, coloured by objective class."""
    plt, _ = setup()
    heldout = heldout or set()
    windows = windows or {"early": (50, 120), "mid": (120, 250), "late": (250, 600)}
    fig = plt.figure(figsize=(TEXT_W, 2.15))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.8, 3.1], wspace=0.3)
    inner = outer[1].subgridspec(1, 3, wspace=0.09)

    ax = fig.add_subplot(outer[0])
    for i, w in enumerate(WIN):
        ax.axvspan(i - 0.42, i + 0.42, color=BAND[w], lw=0, zorder=0)
        b = slopes.get(w) or {}
        if b.get("ci95"):
            lo, hi = b["ci95"]
            ax.plot([i, i], [lo, hi], color=INK, lw=2.4, solid_capstyle="round", zorder=2)
            ax.scatter(i, b["mean"], s=34, color=INK, edgecolor="white", lw=0.9, zorder=3)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(3), [WINDOW_LABEL[w] for w in WIN], fontsize=6.4)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    lim = max(0.03, max(abs(x) for w in WIN for x in (slopes.get(w) or {}).get("ci95") or [0]) * 1.4)
    ax.set(xlim=(-0.6, 2.6), ylim=(-lim, lim), ylabel=r"slope per decade of parameters")
    ygrid(ax)
    title(ax, "a", "Scale")

    by_fam: dict[str, list[dict]] = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(r)
    ylo = min(r[f"{w}_norm"] for r in rows for w in WIN)
    yhi = max(r[f"{w}_norm"] for r in rows for w in WIN)
    axes = []
    for k, w in enumerate(WIN):
        ax = fig.add_subplot(inner[k], sharey=axes[0] if axes else None)
        axes.append(ax)
        for fam, rr in by_fam.items():
            cls = FAMILY_CLASS.get(fam, "supervised")
            rr = sorted(rr, key=lambda r: r["params_m"])
            xs, ys = [r["params_m"] for r in rr], [r[f"{w}_norm"] for r in rr]
            ax.plot(xs, ys, color=tint(CLASS_COLOR[cls], 0.55), lw=0.9, zorder=1)
            for r in rr:
                _mark(ax, r["params_m"], r[f"{w}_norm"], cls, r["model"] in heldout, s=16)
        ax.set(xscale="log", ylim=(ylo - 0.02, yhi + 0.03))
        ax.set_xlabel("parameters (millions)" if k == 1 else "")
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel("normalized alignment")
        ygrid(ax)
        title(ax, "bcd"[k], f"{WINDOW_LABEL[w]} ({windows[w][0]}–{windows[w][1]} ms)")
    from matplotlib.lines import Line2D
    extra = [(Line2D([], [], lw=0, marker="o", ms=4.5, mfc="white", mec=INK2), "development"),
             (Line2D([], [], lw=0, marker="o", ms=4.5, mfc=INK2, mec=INK2), "held-out")]
    present = [c for c in CLASS_ORDER if any(FAMILY_CLASS.get(r["family"]) == c for r in rows)]
    _legend(fig, present, y=-0.2, extra=extra)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: the law, cross-validated on development models and tested on held-out
# ---------------------------------------------------------------------------

def _short(m: str) -> str:
    from .plotting import short_name
    s = short_name(m)
    for a, b in (("hf:microsoft/beit-base-patch16-224-pt22k", "BEiT-B"),
                 ("hf:microsoft/beit-large-patch16-224-pt22k", "BEiT-L"),
                 ("hf:facebook/data2vec-vision-base", "data2vec-B"),
                 ("hf:facebook/data2vec-vision-large", "data2vec-L")):
        if m == a:
            return b
    if m.startswith("eva02_"):
        return "EVA-02-" + ("B" if "base" in m else "L")
    if m.startswith("convnextv2_"):
        return "ConvNeXt V2-" + m.split("_")[1].split(".")[0]
    if "siglip" in m:
        return "SigLIP-" + ("so400m" if "so400m" in m else "B" if "base" in m else "L")
    if m.startswith("aimv2_"):
        return "AIMv2-" + ("H" if "huge" in m else "L")
    if m.endswith(".dino"):
        return "DINO-" + ("S" if "small" in m else "B")
    return s


def fig_law(dev: dict, held: dict, family: dict, test: dict | None, dev_cv: dict | None,
            out_path: str, label: dict | None = None):
    """dev/held: {model: {window: (predicted, observed)}}; test: summary['law_test'];
    dev_cv: law['windows'][w]['cv'] for the cross-validated development fit."""
    plt, _ = setup()
    from .law import _spearman
    label = label or {}
    fig = plt.figure(figsize=(TEXT_W, 2.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.92], wspace=0.36)
    fig.subplots_adjust(right=0.93)
    for k, w in enumerate(WIN):
        ax = fig.add_subplot(gs[k])
        for m, v in dev.items():
            p, o = v[w]
            _mark(ax, p, o, FAMILY_CLASS.get(family.get(m), "supervised"), False, s=14, z=2)
        hp, ho = [], []
        for m, v in held.items():
            p, o = v[w]
            hp.append(p)
            ho.append(o)
            _mark(ax, p, o, FAMILY_CLASS.get(family.get(m), "supervised"), True, s=24, z=4)
            if m in label.get(w, ()):
                _halo(ax.annotate(_short(m), (p, o), xytext=(4, -2), textcoords="offset points",
                                  fontsize=5.6, color=INK2, va="top", zorder=6))
        if len(hp) >= 3:
            c = np.polyfit(hp, ho, 1)
            xx = np.linspace(min(hp), max(hp), 2)
            ax.plot(xx, np.polyval(c, xx), color=INK, lw=0.8, ls=(0, (4, 2)), zorder=3)
        r = (test or {}).get("law_rho", {}).get(w) or {}
        rho = r.get("mean", _spearman(hp, ho) if hp else np.nan)
        ci = r.get("ci95")
        txt = rf"$\rho$ = {rho:+.2f}" + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "")
        ax.text(0.0, 1.02, "held-out " + txt, transform=ax.transAxes, fontsize=6.0,
                color=INK2, va="bottom")
        ax.set_xlabel("predicted (brain-free)" if k == 1 else "")
        if k == 0:
            ax.set_ylabel("observed alignment")
        ygrid(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_title(rf"$\bf{{{'abc'[k]}}}$   {WINDOW_LABEL[w]}", loc="left", pad=13,
                     color=INK, fontsize=8)

    ax = fig.add_subplot(gs[3])
    for i, w in enumerate(WIN):
        ax.axvspan(i - 0.42, i + 0.42, color=BAND[w], lw=0, zorder=0)
        for key, off, col, mk in (("law_rho", -0.12, CLASS_COLOR["latent"], "o"),
                                  ("params_rho", 0.12, MUTED, "s")):
            r = (test or {}).get(key, {}).get(w) or {}
            if r.get("ci95"):
                ax.plot([i + off] * 2, r["ci95"], color=col, lw=2.2, solid_capstyle="round")
                ax.scatter(i + off, r["mean"], s=26, color=col, marker=mk, edgecolor="white",
                           lw=0.8, zorder=3)
        if dev_cv and w in dev_cv:
            ax.scatter(i - 0.12, dev_cv[w]["law"]["spearman"], s=20, marker="o",
                       facecolor="white", edgecolor=CLASS_COLOR["latent"], lw=0.9, zorder=4)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(3), [WINDOW_LABEL[w] for w in WIN], fontsize=6.4)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set(xlim=(-0.6, 2.6), ylim=(-1, 1))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(r"Spearman $\rho$ (predicted vs. observed)")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ygrid(ax)
    ax.set_title(r"$\bf{d}$   Law vs. size", loc="left", pad=13, color=INK, fontsize=8)
    from matplotlib.lines import Line2D
    extra = [(Line2D([], [], lw=0, marker="o", ms=4.5, mfc="white", mec=INK2),
              "development (cross-validated)"),
             (Line2D([], [], lw=0, marker="o", ms=4.5, mfc=INK2, mec=INK2), "held-out (registered)"),
             (Line2D([], [], lw=2.2, color=CLASS_COLOR["latent"]), "law"),
             (Line2D([], [], lw=2.2, color=MUTED), "log params")]
    present = [c for c in CLASS_ORDER
               if any(FAMILY_CLASS.get(family.get(m)) == c for m in list(dev) + list(held))]
    _legend(fig, present, y=-0.17, extra=extra)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: which components carry the law (exploratory)
# ---------------------------------------------------------------------------

def fig_ablation(explo: dict, out_path: str):
    plt, _ = setup()
    names = list(explo["ablation"])
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 1.95), sharey=True,
                             gridspec_kw={"wspace": 0.14})
    ys = np.r_[np.arange(len(names))[::-1] + 1.0, -0.35]          # gap before the last row
    for ax, w in zip(axes, WIN):
        ax.axvline(0, color=INK, lw=0.6)
        for y, n in zip(ys[:-1], names):
            d = explo["ablation"][n][w]
            hero = n == "full law"
            col = CLASS_COLOR["latent"] if hero else INK2
            ax.plot([d["dev_cv"], d["heldout"]], [y, y], color=tint(col, 0.35), lw=1.4, zorder=1)
            ax.scatter(d["dev_cv"], y, s=22, facecolor="white", edgecolor=col, lw=1.0, zorder=2)
            ax.scatter(d["heldout"], y, s=26, color=col, zorder=3)
        nf = explo["without_fcmae"][w]
        ax.plot(nf["ci95"], [ys[-1]] * 2, color=tint(CLASS_COLOR["latent"], 0.6), lw=2.2,
                solid_capstyle="round")
        ax.scatter(nf["rho"], ys[-1], s=26, color=CLASS_COLOR["latent"], zorder=3)
        ax.set(xlim=(-1.05, 1.05), ylim=(-0.95, len(names) + 0.6))
        ax.set_xticks([-1, -0.5, 0, 0.5, 1] if w == "early" else [-0.5, 0, 0.5, 1])
        ax.axhline(0.33, color=GRID, lw=0.8)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        title(ax, "abc"[list(WIN).index(w)], WINDOW_LABEL[w])
        if w == "mid":
            ax.set_xlabel(r"Spearman $\rho$, predicted vs. observed")
    axes[0].set_yticks(ys, names + ["full law without ConvNeXt V2\n(held-out, 95% CI)"],
                       fontsize=6.4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], lw=0, marker="o", ms=4.5, mfc="white", mec=INK2),
         Line2D([], [], lw=0, marker="o", ms=4.5, mfc=INK2, mec=INK2)]
    fig.legend(h, ["development, cross-validated", "held-out"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.55, -0.24), frameon=False, labelcolor=INK2)
    fig.savefig(out_path)
    plt.close(fig)
