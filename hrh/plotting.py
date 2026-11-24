"""Paper figures: one visual system, vector PDF, serif to match the paper.

Encoding rules used everywhere
------------------------------
* family  -> hue (validated categorical order; fixed per family, never cycled)
* size    -> shade of that hue (light = small, dark = large)
* window  -> background band (early / semantic / late), labelled once at the top
* identity is never colour-alone: direct labels or legends + distinct markers
"""
from __future__ import annotations

import numpy as np

# validated categorical palette (light mode) + neutrals
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#ecebe7"
BAND = {"early": "#f4f3f0", "mid": "#e8f0fb", "late": "#f4f3f0"}
NULL = "#b5b4ae"
SEQ = ["#ffffff", "#e6effb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
       "#1c5cab", "#104281", "#0d366b"]
FAMILY_ORDER = ["classic-cnn", "resnet-in1k", "convnext-in22k", "vit-sup", "dinov2", "clip-laion"]
FAMILY_COLOR = {"classic-cnn": SERIES[6], "resnet-in1k": SERIES[1], "convnext-in22k": SERIES[2],
                "vit-sup": SERIES[3], "dinov2": SERIES[0], "clip-laion": SERIES[4]}
FAMILY_LABEL = {"classic-cnn": "AlexNet / VGG", "resnet-in1k": "ResNet",
                "convnext-in22k": "ConvNeXt", "vit-sup": "ViT (supervised)", "dinov2": "DINOv2",
                "clip-laion": "CLIP"}
# objective test (Proposition 3): slots 6, 7, 8 of the validated palette, in an order
# that passes adjacent-pair CVD checks; references are neutral
OBJ_ORDER = ["mae", "vjepa2", "ijepa"]
OBJ_COLOR = {"mae": "#008300", "vjepa2": "#6250d6", "ijepa": "#e34948"}
OBJ_LABEL = {"mae": "MAE", "vjepa2": "V-JEPA 2", "ijepa": "I-JEPA"}
OBJ_MARKER = {"mae": "o", "vjepa2": "s", "ijepa": "D"}
REF_COLOR = "#8a8984"
WINDOW_LABEL = {"early": "early", "mid": "semantic", "late": "late"}
TEXT_W = 6.75


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8,
        "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8, "axes.edgecolor": "#bdbcb6", "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2,
        "ytick.major.size": 2.2, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "lines.linewidth": 1.7, "lines.markersize": 4.2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.6, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "axes.titlelocation": "left",
        "axes.titlepad": 5, "figure.dpi": 150,
    })
    cmap = LinearSegmentedColormap.from_list("hrh_seq", SEQ)
    return plt, cmap


def tint(hex_color: str, f: float) -> str:
    """Mix a colour with white: f=1 -> colour, f=0 -> white."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    m = 255 - (255 - c) * f
    return "#" + "".join(f"{int(round(v)):02x}" for v in m)


def shade(hex_color: str, f: float) -> str:
    """f in [0,1]: 0 -> light tint, 1 -> slightly darkened colour (for 'large')."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    light = 255 - (255 - c) * 0.32
    dark = c * 0.78
    mid = c
    m = light + (mid - light) * min(f / 0.7, 1) if f <= 0.7 else mid + (dark - mid) * (f - 0.7) / 0.3
    return "#" + "".join(f"{int(round(v)):02x}" for v in m)


def shades(hex_color: str, n: int) -> list[str]:
    return [shade(hex_color, f) for f in np.linspace(0.0, 1.0, n)] if n > 1 else [hex_color]


def title(ax, letter: str, text: str):
    ax.set_title((rf"$\bf{{{letter}}}$" + "   " if letter else "") + text, loc="left",
                 color=INK, fontsize=8)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def window_bands(ax, windows, labels=True):
    from matplotlib.transforms import blended_transform_factory
    trf = blended_transform_factory(ax.transData, ax.transAxes)
    for k, (w, (a, b)) in enumerate(windows.items()):
        ax.axvspan(a, b, color=BAND[w], lw=0, zorder=0)
        if labels:                                   # stagger so narrow bands never collide
            ax.text((a + b) / 2, 0.985 - 0.075 * (k % 2), WINDOW_LABEL[w], transform=trf,
                    ha="center", va="top", fontsize=6.1, color=MUTED, style="italic")


def onset(ax):
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1)


def short_name(m: str) -> str:
    """resnet50.tv_in1k -> ResNet-50, vit_large_patch14_dinov2 -> DINOv2-L/14, ..."""
    import re
    if m.startswith("hf:"):                          # hf:facebook/ijepa_vith14_22k etc.
        tail = m.split("/")[-1]
        mm = re.match(r"(ijepa|vjepa2)[-_]vit([lhg])(\d*)", tail)
        if mm:
            kind = "I-JEPA" if mm.group(1) == "ijepa" else "V-JEPA 2"
            sz = {"l": "L", "h": "H", "g": "g"}[mm.group(2)]
            return f"{kind}-{sz}/{mm.group(3) or 16}"
        return tail
    base = m.split(".")[0].replace("tv:", "")
    size = {"tiny": "Ti", "small": "S", "base": "B", "large": "L", "huge": "H", "giant": "g",
            "gigantic": "bigG", "xlarge": "XL"}
    if base.startswith("resnet"):
        return "ResNet-" + base[6:]
    if base.startswith("convnext_"):
        k = base.split("_")[1]
        return "ConvNeXt-" + (size[k] if k == "xlarge" else size[k][0])
    if base.startswith("vgg"):
        return "VGG-" + base[3:]
    if base == "alexnet":
        return "AlexNet"
    mm = re.match(r"vit_(\w+?)_patch(\d+)", base)
    if mm:
        tag = f"{size.get(mm.group(1), mm.group(1))}/{mm.group(2)}"
        if "dinov2" in base:
            return "DINOv2-" + tag
        if "clip" in base:
            return "CLIP-" + tag
        if m.endswith(".mae"):
            return "MAE-" + tag
        return "ViT-" + tag
    return base


def ridge(A: np.ndarray, cols: np.ndarray, smooth: int = 5, frac: float = 0.3) -> np.ndarray:
    """Alignment-weighted mean relative depth per time bin ("where in the network the
    brain is, right now"), shown only on `cols` where alignment is substantial."""
    w = np.clip(A, 0, None) ** 2
    depth = np.linspace(0, 1, len(A))[:, None]
    with np.errstate(all="ignore"):
        d = (w * depth).sum(0) / w.sum(0)
    d = np.nan_to_num(d, nan=0.5)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        d = np.convolve(np.pad(d, smooth // 2, mode="edge"), k, mode="valid")[: len(d)]
    strong = A.max(0) > frac * A.max()
    return np.where(cols & strong, d, np.nan)


# ---------------------------------------------------------------------------
# Theory figure
# ---------------------------------------------------------------------------

def fig_theory(out_path: str) -> dict:
    from . import theory as th
    plt, _ = setup()
    prob = th.make_problem()
    s = th.train(prob, steps=15000)
    m = th.train(prob, steps=15000, recon=1.0)
    c = s["ceiling_early"]
    blue, orange = SERIES[0], SERIES[1]

    fig = plt.figure(figsize=(TEXT_W, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.38)

    # (a) early alignment over training
    ax = fig.add_subplot(gs[0])
    x = s["step"] + 1
    ax.axhspan(0, c, color="#f4f3f0", lw=0, zorder=0)
    ax.axhline(c, color=INK, lw=0.8, ls=(0, (4, 2)))
    ax.text(1.25, c - 0.02, r"ceiling $\rho_S$ for single-readout nets (Thm. 1)", fontsize=6.3,
            color=INK2, va="top")
    ax.plot(x, m["cka_early_l1"], color=orange, lw=1.9)
    ax.plot(x, s["cka_early_l1"], color=blue, lw=1.9)
    ax.text(x[-1] * 0.9, m["cka_early_l1"][-1] + 0.035, "+ reconstruction readout",
            ha="right", fontsize=6.6, color=orange)
    ax.text(1.3, s["cka_early_l1"][0] - 0.045, "output readout only", ha="left",
            fontsize=6.6, color=blue)
    ax.annotate("scaling inversion", xy=(3500, 0.36), xytext=(9, 0.40), fontsize=6.4,
                color=INK2, ha="left", arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7,
                                                       connectionstyle="arc3,rad=-0.25"))
    ax.set(xscale="log", xlim=(1, x[-1] * 1.3), ylim=(0, 0.78), xlabel="optimization steps",
           ylabel="alignment with early stage (CKA)")
    ax.spines["bottom"].set_bounds(1, x[-1])
    title(ax, "a", "Early alignment falls to the ceiling")

    # (b) layer profile: the trajectory collapses
    ax = fig.add_subplot(gs[1])
    L = len(s["ws"])
    depth = np.arange(1, L + 1)

    def prof(ws, G):
        return [th.cos(g, G) for g in th.grams(ws)]
    ax.axhline(c, color=INK, lw=0.8, ls=(0, (4, 2)))
    p_init = prof(s["ws_init"], prob["G_early"])
    p_tr = prof(s["ws"], prob["G_early"])
    p_sem = prof(s["ws"], prob["G_mid"])
    p_rec = prof(m["ws"], prob["G_early"])
    ax.plot(depth, p_init, color=tint(blue, 0.45), marker="o", ls=(0, (3, 1.5)), mfc="white",
            mew=1.1)
    ax.plot(depth, p_tr, color=blue, marker="o")
    ax.plot(depth, p_sem, color=INK, marker="s", ms=3.6)
    ax.plot(depth, p_rec, color=orange, marker="^")
    lab = dict(fontsize=6.4, va="center")
    ax.text(L + 0.15, p_sem[-1], "semantic, trained", color=INK, **lab)
    ax.text(L + 0.15, p_init[-1] + 0.02, "early, init", color=INK2, **lab)
    ax.annotate("early, + recon.\n(readout at layer 1)", xy=(1, p_rec[0]), xytext=(1.25, 0.72),
                fontsize=6.4, color=orange, va="bottom")
    ax.text(L + 0.15, p_tr[-1] - 0.035, "early, trained", color=blue, **lab)
    ax.set(xticks=depth, xlim=(0.7, L + 2.3), ylim=(0, 1.05), xlabel="layer",
           ylabel="CKA with brain-like stage")
    ax.spines["bottom"].set_bounds(1, L)
    title(ax, "b", "Every layer collapses")

    # (c) erasure + power collapse
    ax = fig.add_subplot(gs[2])
    ax.plot(x, s["nuisance_l1"] + 1e-7, color=tint(blue, 0.55), lw=3.4)
    ax.plot(x, s["collapse_err"] + 1e-7, color=SERIES[2], lw=1.3, ls=(0, (4, 1.6)))
    ax.text(1.4, 0.3, "nuisance in layer 1\n(Prop. 1)", fontsize=6.4, color=blue, va="top")
    ax.text(25, 1.5e-2, r"$\max_t \|G_t - G_T^{t/T}\|$" + "\n(Prop. 2)", fontsize=6.4,
            color=SERIES[2], va="top")
    ax.set(xscale="log", yscale="log", ylim=(5e-6, 3), xlabel="optimization steps",
           ylabel="relative error")
    ygrid(ax)
    title(ax, "c", "Erasure and power collapse")
    fig.savefig(out_path)
    plt.close(fig)
    return {"ceiling": c, "early_init": float(s["cka_early_l1"][0]),
            "early_single": float(s["cka_early_l1"][-1]),
            "early_multi": float(m["cka_early_l1"][-1]),
            "mid_single": float(s["cka_mid_top"][-1]), "mid_multi": float(m["cka_mid_top"][-1]),
            "nuisance_final": float(s["nuisance_l1"][-1]),
            "collapse_final": float(s["collapse_err"][-1])}


# ---------------------------------------------------------------------------
# Validation figure
# ---------------------------------------------------------------------------

def fig_validation(out_path: str, seed: int = 1) -> dict:
    from . import brain, synthetic as syn, trajectory as tr
    plt, cmap = setup()
    rng = np.random.default_rng(seed)
    f = syn.factors(rng, 120)
    subj = brain.subject_rdms("synth", syn.eeg(rng, f), syn.TIMES, n_splits=4, n_ceiling_splits=3)
    t = syn.TIMES
    mask = tr.reliable_mask(t, subj.split_half_r)
    kinds = {"hier": ("trajectory", SERIES[0], "o"), "static": ("bottleneck", SERIES[1], "s"),
             "noise": ("null", NULL, "^")}
    res = {}
    for k in kinds:
        rd = syn.model_rdms(rng, f, k)
        a = tr.alignment_matrix(rd, subj.full)
        an = tr.normalize(a, subj.ceiling, mask)
        res[k] = dict(A=a, cov=tr.coverage_auc(an), dtw=tr.dtw_distance(an),
                      peak=float(np.nanmax(an)), mid=tr.window_scores(rd, subj)["mid"]["norm"])

    fig = plt.figure(figsize=(TEXT_W, 2.0))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.05, 1, 1, 0.045, 0.62, 0.95], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax.fill_between(t, 0, subj.ceiling, where=mask, color=tint(SERIES[0], 0.25), lw=0,
                    step="mid", zorder=1)
    ax.plot(t, subj.ceiling, color=INK, lw=1.1, zorder=2)
    onset(ax)
    ax.set(xlim=(-200, 790), ylim=(0, 1.12), xlabel="time from onset (ms)", ylabel="noise ceiling",
           yticks=[0, 0.5, 1])
    ax.text(235, 0.42, "reliable\nbins", ha="center", fontsize=6.3, color=SERIES[0])
    title(ax, "a", "Signal vs. noise")

    for i, k in enumerate(("hier", "static")):
        ax = fig.add_subplot(gs[1 + i])
        A = res[k]["A"]
        im = ax.imshow(np.clip(A, 0, None), aspect="auto", origin="lower", cmap=cmap,
                       extent=[t[0], t[-1], 0, 1], vmin=0, vmax=0.85, interpolation="bilinear")
        r = ridge(A, mask)
        ax.plot(t, r, color="white", lw=2.6)
        ax.plot(t, r, color=INK, lw=0.9)
        onset(ax)
        ax.set(xlabel="time from onset (ms)", yticks=[0, 1], yticklabels=["in", "out"])
        if i == 0:
            ax.set_ylabel("model depth", labelpad=-2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        title(ax, "bc"[i], f"{kinds[k][0].capitalize()} model")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[3]), ticks=[0, 0.4, 0.8])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.ax.set_title("RSA", fontsize=6.3, pad=3, color=INK2)

    # (d) dumbbell: same peak, different trajectory
    ax = fig.add_subplot(gs[5])
    rows = [("semantic score", "mid"), ("coverage", "cov"), (r"$1 - D_{\mathrm{HRH}}$", "dtw")]
    for j, (lab, key) in enumerate(rows):
        y = len(rows) - 1 - j
        vals = {k: (1 - res[k][key] if key == "dtw" else res[k][key]) for k in kinds}
        ax.plot([vals["static"], vals["hier"]], [y, y], color="#dcdbd6", lw=3.4, zorder=1)
        for k, (_n, col, mk) in kinds.items():
            ax.scatter(vals[k], y, s=34, color=col, zorder=3, edgecolor="white", lw=0.8,
                       marker=mk)
        ax.text(-0.06, y, lab, ha="right", va="center", fontsize=6.8, color=INK2)
    ax.set(xlim=(-0.02, 1.0), ylim=(-0.65, len(rows) - 0.3), yticks=[], xticks=[0, 0.5, 1])
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    for k, (name, col, mk) in kinds.items():
        ax.scatter([], [], color=col, s=24, label=name, marker=mk)
    ax.legend(loc="upper center", bbox_to_anchor=(0.25, -0.13), ncol=3, columnspacing=0.8,
              handletextpad=0.2)
    ax.set_title(r"$\bf{d}$   Same peak, different trajectory", loc="right", fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)
    return {k: {kk: v[kk] for kk in ("cov", "dtw", "peak", "mid")} for k, v in res.items()}


# ---------------------------------------------------------------------------
# Results figures
# ---------------------------------------------------------------------------

def _families(rows):
    return [f for f in FAMILY_ORDER if any(r["family"] == f for r in rows)]


def fig_windows(rows: list[dict], curves: dict, times: np.ndarray, windows: dict,
                out_path: str):
    """Hero: best-layer alignment over brain time, one panel per family, shade = size."""
    plt, _ = setup()
    fams = _families(rows)
    ncol = 3 if len(fams) > 4 else len(fams)
    nrow = int(np.ceil(len(fams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXT_W, 1.62 * nrow + 0.1), sharex=True,
                             sharey=True, squeeze=False,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.1})
    vmax = np.nanmax([np.nanmax(c) for c in curves.values() if np.isfinite(c).any()])
    for ax, f in zip(axes.flat, fams):
        rr = sorted([r for r in rows if r["family"] == f], key=lambda r: r["params_m"])
        cols = shades(FAMILY_COLOR[f], len(rr))
        window_bands(ax, windows, labels=(ax in axes[0]))
        onset(ax)
        for r, col in zip(rr, cols):
            ax.plot(times, curves[r["model"]], color=col, lw=1.5)
        for j, (r, col) in enumerate(zip(rr[::-1], cols[::-1])):
            y = 0.76 - 0.1 * j
            ax.plot([0.70, 0.75], [y, y], transform=ax.transAxes, color=col, lw=2.2,
                    solid_capstyle="butt")
            ax.text(0.765, y, short_name(r["model"]), transform=ax.transAxes, ha="left",
                    va="center", fontsize=5.8, color=INK2)
        ax.set_title(FAMILY_LABEL[f], loc="left", fontsize=7.8, color=FAMILY_COLOR[f],
                     fontweight="bold")
        ax.set(xlim=(-100, 700), ylim=(-0.05, vmax * 1.12))
    for ax in list(axes.flat)[len(fams):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time from stimulus onset (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("normalized alignment")
    fig.savefig(out_path)
    plt.close(fig)


def fig_scaling(rows: list[dict], windows: dict, out_path: str, boots: dict | None = None,
                summary: dict | None = None):
    """(a) scaling slope per window with CI; (b) coverage and (c) D_HRH vs scale."""
    plt, _ = setup()
    fams = _families(rows)
    fig = plt.figure(figsize=(TEXT_W, 2.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    ws = list(windows)
    for i, w in enumerate(ws):
        ax.axvspan(i - 0.42, i + 0.42, color=BAND[w], lw=0, zorder=0)
        b = (boots or {}).get(w) or {}
        if b.get("ci95"):
            lo, hi = b["ci95"]
            col = SERIES[0] if b["mean"] > 0 else SERIES[1]
            ax.plot([i, i], [lo, hi], color=col, lw=2.6, zorder=2)
            ax.scatter(i, b["mean"], s=46, color=col, edgecolor="white", lw=1.0, zorder=3)
            ax.text(i + 0.12, b["mean"], f"{b['mean']:+.2f}", fontsize=6.4, va="center",
                    color=INK2)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(range(len(ws)), [f"{WINDOW_LABEL[w]}\n{windows[w][0]}–{windows[w][1]}"
                                   for w in ws], fontsize=6.3)
    ax.tick_params(axis="x", length=0)
    ax.set(xlim=(-0.6, len(ws) - 0.4), ylabel=r"scaling slope $\beta$ (per decade)")
    ax.set_xlabel("window (ms)")
    ax.spines["bottom"].set_visible(False)
    ygrid(ax)
    title(ax, "a", "Scaling slope by window")

    for k, (key, lab, let) in enumerate(
            (("coverage_auc", "temporal coverage", "b"),
             ("D_HRH", r"trajectory distance $D_{\mathrm{HRH}}$", "c"))):
        ax = fig.add_subplot(gs[1 + k])
        for f in fams:
            rr = sorted([r for r in rows if r["family"] == f], key=lambda r: r["params_m"])
            col = FAMILY_COLOR[f]
            i = FAMILY_ORDER.index(f)
            xs, ys = [r["params_m"] for r in rr], [r[key] for r in rr]
            ax.plot(xs, ys, color=col, lw=1.4, zorder=2)
            ax.scatter(xs, ys, c=shades(col, len(rr)), marker=MARKERS[i], s=22,
                       edgecolor="white", lw=0.6, zorder=3, label=FAMILY_LABEL[f])
        ax.set(xscale="log", xlabel="parameters (millions)", ylabel=lab)
        ygrid(ax)
        slope = (summary or {}).get(f"{key}_slope")
        name = "Coverage" if k == 0 else "Trajectory distance"
        title(ax, let, name + (rf"   $\beta$ = {slope:+.2f}" if slope is not None else ""))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=FAMILY_COLOR[f], marker=MARKERS[FAMILY_ORDER.index(f)], lw=1.4,
                ms=4.5, mec="white", mew=0.6) for f in fams]
    fig.legend(h, [FAMILY_LABEL[f] for f in fams], loc="lower center", ncol=len(fams),
               bbox_to_anchor=(0.62, -0.2), columnspacing=1.4, handletextpad=0.4)
    fig.savefig(out_path)
    plt.close(fig)


def fig_layer_time(mats: dict, times: np.ndarray, params: dict, family: dict, out_path: str,
                   common_mask: np.ndarray | None = None):
    """Smallest vs. largest model of every family, with the best-layer ridge."""
    plt, cmap = setup()
    fams = [f for f in FAMILY_ORDER if any(family[m] == f for m in mats)]
    fig, axes = plt.subplots(2, len(fams), figsize=(TEXT_W, 2.55), squeeze=False, sharex=True,
                             gridspec_kw={"hspace": 0.32, "wspace": 0.12})
    vmax = np.percentile(np.concatenate([a.ravel() for a in mats.values()]), 99.5)
    cols = common_mask if common_mask is not None else times >= 0
    im = None
    for j, f in enumerate(fams):
        ms = sorted([m for m in mats if family[m] == f], key=lambda m: params[m])
        for i, m in enumerate((ms[0], ms[-1])):
            ax = axes[i, j]
            A = mats[m]
            im = ax.imshow(np.clip(A, 0, None), aspect="auto", origin="lower", cmap=cmap,
                           vmin=0, vmax=vmax, extent=[times[0], times[-1], 0, 1],
                           interpolation="bilinear")
            r = ridge(A, cols)
            ax.plot(times, r, color="white", lw=2.2)
            ax.plot(times, r, color=INK, lw=0.7)
            onset(ax)
            ax.set(yticks=[], xlim=(-100, 700))
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(short_name(m), loc="left", fontsize=6.6,
                         color=FAMILY_COLOR[f] if i == 0 else INK)
            if j == 0:
                ax.set_ylabel(("smallest" if i == 0 else "largest") + "\ndepth →", fontsize=6.6)
            if i == 1:
                ax.set_xlabel("ms", fontsize=6.6)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.ax.set_title("RSA", fontsize=6.3, color=INK2)
    fig.savefig(out_path)
    plt.close(fig)


def fig_ceiling(subjects: list, out_path: str, masks: dict, windows: dict):
    plt, _ = setup()
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    t = subjects[0].times_ms
    c = np.stack([s.ceiling for s in subjects])
    common = np.all(np.stack([masks[s.subject] for s in subjects]), 0)
    window_bands(ax, windows)
    ax.fill_between(t, 0, c.mean(0), where=common, color=tint(SERIES[0], 0.25), lw=0, step="mid")
    for row in c:
        ax.plot(t, row, color=tint(SERIES[0], 0.55), lw=0.5)
    ax.plot(t, c.mean(0), color=INK, lw=1.3)
    onset(ax)
    ax.set(xlabel="time from stimulus onset (ms)", ylabel="noise ceiling", ylim=(0, 1.1),
           xlim=(t[0], t[-1]), yticks=[0, 0.5, 1])
    fig.savefig(out_path)
    plt.close(fig)


def fig_objective(rows: list[dict], test: dict, windows: dict, size_class: dict,
                  references: dict, out_path: str):
    """(a-c) each objective model against the mean of its size-matched single-readout
    references, per window; (d) the pre-registered contrast with 95% CIs."""
    plt, _ = setup()
    by = {r["model"]: r for r in rows}
    members = [(f, m) for f in OBJ_ORDER for m in test["members"].get(f, []) if m in by]
    members.sort(key=lambda fm: (OBJ_ORDER.index(fm[0]), by[fm[1]]["params_m"]))
    fig = plt.figure(figsize=(TEXT_W, 0.32 * len(members) + 1.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.12)
    ws = list(windows)
    ys = np.arange(len(members))[::-1]
    lo_all = min(min(by[m][f"{w}_norm"] for _f, m in members) for w in ws)
    hi_all = max(max(by[m][f"{w}_norm"] for _f, m in members) for w in ws)
    for k, w in enumerate(ws):
        ax = fig.add_subplot(gs[k])
        ax.axvspan(-1, 2, color=BAND[w], lw=0, zorder=0)
        for y, (f, m) in zip(ys, members):
            refs = [by[r][f"{w}_norm"] for r in references.get(size_class.get(m), []) if r in by]
            x = by[m][f"{w}_norm"]
            if refs:
                xr = float(np.mean(refs))
                lo_all, hi_all = min(lo_all, xr), max(hi_all, xr)
                ax.plot([xr, x], [y, y], color=tint(OBJ_COLOR[f], 0.55), lw=1.6, zorder=2)
                ax.scatter(xr, y, s=20, facecolor="white", edgecolor=REF_COLOR, lw=1.0, zorder=3)
            ax.scatter(x, y, s=26, color=OBJ_COLOR[f], marker=OBJ_MARKER[f],
                       edgecolor="white", lw=0.6, zorder=4)
        ax.set_yticks(ys, [short_name(m) for _f, m in members] if k == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(members) - 0.3)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        title(ax, "abc"[k], f"{WINDOW_LABEL[w]} ({windows[w][0]}–{windows[w][1]} ms)")
        if k == 1:
            ax.set_xlabel("normalized alignment")
    pad = 0.08 * (hi_all - lo_all)
    for ax in fig.axes:
        ax.set_xlim(lo_all - pad, hi_all + pad)
    for lab in fig.axes[0].get_yticklabels():
        for f, m in members:
            if lab.get_text() == short_name(m):
                lab.set_color(OBJ_COLOR[f])

    ax = fig.add_subplot(gs[3])
    series = [("all", INK, test["contrast"])] + [
        (f, OBJ_COLOR[f], test["by_family"][f]) for f in OBJ_ORDER if f in test["by_family"]]
    off = np.linspace(-0.27, 0.27, len(series))
    for i, w in enumerate(ws):
        ax.axvspan(i - 0.42, i + 0.42, color=BAND[w], lw=0, zorder=0)
        for (f, col, d), o in zip(series, off):
            c = d[w]
            if not c.get("ci95"):
                continue
            big = f == "all"
            ax.plot([i + o] * 2, c["ci95"], color=col, lw=2.6 if big else 1.4, zorder=2)
            ax.scatter(i + o, c["mean"], s=40 if big else 18, color=col,
                       marker="o" if big else OBJ_MARKER[f], edgecolor="white",
                       lw=0.8, zorder=3)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(range(len(ws)), [WINDOW_LABEL[w] for w in ws])
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set(xlim=(-0.6, len(ws) - 0.4), ylabel=r"$\Delta$ vs. size-matched references (RSA)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ygrid(ax)
    title(ax, "d", "Objective contrast")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=OBJ_COLOR[f], marker=OBJ_MARKER[f], lw=0, ms=4.5, mec="white")
         for f in OBJ_ORDER if f in test["by_family"]]
    labs = [OBJ_LABEL[f] for f in OBJ_ORDER if f in test["by_family"]]
    h += [Line2D([], [], color=INK, marker="o", lw=0, ms=5.5, mec="white"),
          Line2D([], [], color=REF_COLOR, marker="o", mfc="white", lw=0, ms=4.5)]
    labs += ["all objective models", "size-matched reference (ViT / CLIP)"]
    fig.legend(h, labs, loc="lower center", ncol=len(labs), bbox_to_anchor=(0.5, -0.1),
               columnspacing=1.3, handletextpad=0.4)
    fig.savefig(out_path)
    plt.close(fig)
