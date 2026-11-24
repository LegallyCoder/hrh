"""Controlled objective experiment: one architecture, one dataset, one budget;
only the training objective changes.

  sup       supervised classification (1000 ImageNet classes)
  mae       masked autoencoding: reconstruct normalized pixels of masked patches
  jepa      I-JEPA-style: predict EMA-target embeddings of masked blocks from a context block
  data2vec  predict the EMA teacher's averaged, normalized top-layer features at masked positions

Shared: ViT-Tiny (dim 192, depth 12, 3 heads) on 64x64 images with 8x8 patches
(an 8x8 token grid, no class token), AdamW, cosine schedule, identical GPU
augmentation (random resized crop + flip), identical number of images seen.
Data: ImageNet-1k downsampled to 64x64 (Hugging Face benjamin-paine/imagenet-1k-64x64),
decoded once per session into a uint8 memmap on local disk.
Checkpoints go to `out_dir` (Google Drive) and runs resume automatically.
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import time

import numpy as np

OBJECTIVES = ("sup", "mae", "jepa", "data2vec")
IMG, PATCH = 64, 8
GRID = IMG // PATCH
DIM, DEPTH, HEADS = 192, 12, 3
HF_DATASET = "benjamin-paine/imagenet-1k-64x64"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data(cache_dir: str, split: str = "train", limit: int | None = None) -> tuple[str, str]:
    """Decode the parquet shards into (N, 64, 64, 3) uint8 memmap + int16 labels."""
    os.makedirs(cache_dir, exist_ok=True)
    xp, yp = (os.path.join(cache_dir, f"{split}_x.u8"), os.path.join(cache_dir, f"{split}_y.npy"))
    if os.path.exists(xp) and os.path.exists(yp):
        return xp, yp
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    from PIL import Image
    root = snapshot_download(HF_DATASET, repo_type="dataset", allow_patterns=[f"data/{split}-*"])
    files = sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    n = min(n, limit) if limit else n
    x = np.lib.format.open_memmap(xp + ".npy", mode="w+", dtype=np.uint8, shape=(n, IMG, IMG, 3))
    y = np.zeros(n, np.int16)
    i = 0
    for f in files:
        tab = pq.read_table(f)
        cols = tab.column_names
        icol = "image" if "image" in cols else cols[0]
        lcol = "label" if "label" in cols else cols[1]
        for img, lab in zip(tab.column(icol).to_pylist(), tab.column(lcol).to_pylist()):
            if i >= n:
                break
            b = img["bytes"] if isinstance(img, dict) else img
            im = Image.open(io.BytesIO(b)).convert("RGB")
            if im.size != (IMG, IMG):
                im = im.resize((IMG, IMG), Image.BICUBIC)
            x[i] = np.asarray(im)
            y[i] = lab
            i += 1
        print(f"[data] {split}: {i}/{n}", flush=True)
        if i >= n:
            break
    x.flush()
    os.replace(xp + ".npy", xp)
    np.save(yp, y[:i])
    return xp, yp


def load_data(xp: str, yp: str):
    y = np.load(yp)
    x = np.load(xp, mmap_mode="r") if xp.endswith(".npy") else np.lib.format.open_memmap(xp, mode="r")
    return x, y


class BufferedBatches:
    """Fast shuffled batches from a large on-disk memmap.

    Random access to single images is disk-bound, so a background thread reads
    `n_chunks` contiguous chunks from random offsets (sequential reads), shuffles their
    union in memory and yields batches; the next buffer is loaded while the current
    one is consumed. Each buffer mixes images from many parts of the dataset."""

    def __init__(self, x, y, batch: int, seed: int, chunk: int = 8192, n_chunks: int = 8):
        import queue
        import threading
        self.x, self.y, self.batch = x, y, batch
        self.chunk, self.n_chunks = min(chunk, len(y)), n_chunks
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.stop = False
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def _load(self):
        n = len(self.y)
        starts = self.rng.integers(0, max(1, n - self.chunk), self.n_chunks)
        xs = [np.asarray(self.x[s:s + self.chunk]) for s in starts]       # sequential reads
        ys = [np.asarray(self.y[s:s + self.chunk]) for s in starts]
        xb, yb = np.concatenate(xs), np.concatenate(ys).astype(np.int64)
        perm = self.rng.permutation(len(yb))
        return xb[perm], yb[perm]

    def _work(self):
        while not self.stop:
            self.q.put(self._load())

    def __iter__(self):
        while True:
            xb, yb = self.q.get()
            for i in range(0, len(yb) - self.batch + 1, self.batch):
                yield xb[i:i + self.batch], yb[i:i + self.batch]


def gpu_augment(xb, gen=None):
    """uint8 (B, 64, 64, 3) on GPU -> normalized float (B, 3, 64, 64) with random resized
    crop (scale 0.25-1, ratio 3/4-4/3) and horizontal flip, via one affine grid."""
    import torch
    import torch.nn.functional as F
    b = xb.shape[0]
    x = xb.permute(0, 3, 1, 2).float() / 255.0
    dev = x.device
    scale = torch.empty(b, device=dev).uniform_(0.25, 1.0, generator=gen)
    logr = torch.empty(b, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3), generator=gen)
    r = torch.exp(logr)
    sw = torch.sqrt(scale * r).clamp(max=1.0)
    sh = torch.sqrt(scale / r).clamp(max=1.0)
    tx = (torch.rand(b, device=dev, generator=gen) * 2 - 1) * (1 - sw)
    ty = (torch.rand(b, device=dev, generator=gen) * 2 - 1) * (1 - sh)
    flip = torch.where(torch.rand(b, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = sw * flip
    theta[:, 1, 1] = sh
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (b, 3, IMG, IMG), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    m = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - m) / s


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _blocks(dim, depth, heads):
    import torch.nn as nn
    from timm.models.vision_transformer import Block
    return nn.ModuleList([Block(dim, heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(depth)])


def build_encoder():
    import torch
    import torch.nn as nn

    class Encoder(nn.Module):
        """Plain ViT without class token; `blocks` is hooked for feature extraction."""

        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, DIM, PATCH, PATCH)
            self.pos_embed = nn.Parameter(torch.zeros(1, GRID * GRID, DIM))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.blocks = _blocks(DIM, DEPTH, HEADS)
            self.norm = nn.LayerNorm(DIM)
            self.num_prefix_tokens = 0

        def tokens(self, x):
            return self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed

        def forward_tokens(self, t, keep=None, return_layers=False):
            if keep is not None:                                   # (B, k) visible indices
                t = torch.gather(t, 1, keep[..., None].expand(-1, -1, t.shape[-1]))
            layers = []
            for blk in self.blocks:
                t = blk(t)
                if return_layers:
                    layers.append(t)
            t = self.norm(t)
            return (t, layers) if return_layers else t

        def forward(self, x):                                      # pooled embedding
            return self.forward_tokens(self.tokens(x)).mean(1)

    return Encoder()


def build(objective: str):
    """Returns (model, loss_fn(model, x, y) -> loss, ema_pairs)."""
    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    enc = build_encoder()
    n = GRID * GRID

    if objective == "sup":
        head = nn.Linear(DIM, 1000)
        model = nn.ModuleDict({"enc": enc, "head": head})

        def loss(m, x, y):
            return F.cross_entropy(m["head"](m["enc"](x)), y, label_smoothing=0.1)
        return model, loss, []

    if objective == "mae":
        ddim = 128
        model = nn.ModuleDict({"enc": enc, "proj": nn.Linear(DIM, ddim),
                               "dec": _blocks(ddim, 4, 4), "dnorm": nn.LayerNorm(ddim),
                               "pred": nn.Linear(ddim, PATCH * PATCH * 3)})
        model.mask_token = nn.Parameter(torch.zeros(1, 1, ddim))
        model.dpos = nn.Parameter(torch.zeros(1, n, ddim))
        nn.init.trunc_normal_(model.dpos, std=0.02)

        def loss(m, x, y):
            b = x.shape[0]
            k = n // 4                                             # 75% masked
            perm = torch.rand(b, n, device=x.device).argsort(1)
            keep, drop = perm[:, :k], perm[:, k:]
            z = m["proj"](m["enc"].forward_tokens(m["enc"].tokens(x), keep))
            full = m.mask_token.to(z.dtype).expand(b, n, -1).clone()
            full.scatter_(1, keep[..., None].expand(-1, -1, z.shape[-1]), z)
            h = full + m.dpos
            for blk in m["dec"]:
                h = blk(h)
            pred = m["pred"](m["dnorm"](h))
            p = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)          # (B,3,G,G,P,P)
            p = p.permute(0, 2, 3, 1, 4, 5).reshape(b, n, -1)
            p = (p - p.mean(-1, keepdim=True)) / (p.var(-1, keepdim=True) + 1e-6).sqrt()
            idx = drop[..., None].expand(-1, -1, p.shape[-1])
            return F.mse_loss(torch.gather(pred, 1, idx), torch.gather(p, 1, idx))
        return model, loss, []

    if objective == "jepa":
        pdim = 96
        target = copy.deepcopy(enc).requires_grad_(False)
        model = nn.ModuleDict({"enc": enc, "target": target, "pin": nn.Linear(DIM, pdim),
                               "pred": _blocks(pdim, 4, 3), "pnorm": nn.LayerNorm(pdim),
                               "pout": nn.Linear(pdim, DIM)})
        model.mask_token = nn.Parameter(torch.zeros(1, 1, pdim))
        model.ppos = nn.Parameter(torch.zeros(1, n, pdim))
        nn.init.trunc_normal_(model.ppos, std=0.02)

        def blocks_mask(b, dev):
            """4 target blocks (15-20% of the grid each); context = the rest of a large block."""
            tgt = np.zeros((b, GRID, GRID), dtype=bool)          # built on CPU, moved once
            for _ in range(4):
                area = np.random.uniform(0.15, 0.2, b) * n
                ar = np.exp(np.random.uniform(math.log(0.75), math.log(1.5), b))
                hh = np.clip(np.round(np.sqrt(area * ar)), 1, GRID).astype(int)
                ww = np.clip(np.round(np.sqrt(area / ar)), 1, GRID).astype(int)
                r0 = np.random.randint(0, GRID - hh + 1)
                c0 = np.random.randint(0, GRID - ww + 1)
                for i in range(b):
                    tgt[i, r0[i]:r0[i] + hh[i], c0[i]:c0[i] + ww[i]] = True
            return torch.from_numpy(tgt.reshape(b, -1)).to(dev)

        def loss(m, x, y):
            b = x.shape[0]
            tgt = blocks_mask(b, x.device)
            with torch.no_grad():
                t = m["target"].forward_tokens(m["target"].tokens(x))
                t = F.layer_norm(t, (t.shape[-1],))
            # context: non-target tokens, padded to a common length by repeating
            ctx_n = int((~tgt).sum(1).min().item())
            score = torch.rand(b, n, device=x.device) + tgt.float() * 2.0   # targets last
            keep = score.argsort(1)[:, :ctx_n]
            z = m["pin"](m["enc"].forward_tokens(m["enc"].tokens(x), keep))
            full = m.mask_token.to(z.dtype).expand(b, n, -1).clone()
            full.scatter_(1, keep[..., None].expand(-1, -1, z.shape[-1]), z)
            h = full + m.ppos
            for blk in m["pred"]:
                h = blk(h)
            pred = m["pout"](m["pnorm"](h)).float()
            w = tgt.float()[..., None]
            return ((pred - t.float()) ** 2 * w).sum() / (w.sum() * DIM)
        return model, loss, [("enc", "target")]

    if objective == "data2vec":
        teacher = copy.deepcopy(enc).requires_grad_(False)
        model = nn.ModuleDict({"enc": enc, "teacher": teacher, "reg": nn.Linear(DIM, DIM)})
        model.mask_token = nn.Parameter(torch.zeros(1, 1, DIM))
        top_k = 4

        def loss(m, x, y):
            b = x.shape[0]
            with torch.no_grad():
                _, layers = m["teacher"].forward_tokens(m["teacher"].tokens(x), return_layers=True)
                tgt = sum(F.layer_norm(h, (h.shape[-1],)) for h in layers[-top_k:]) / top_k
            tok = m["enc"].tokens(x)
            mask = torch.rand(b, n, device=x.device) < 0.6
            tok = torch.where(mask[..., None], m.mask_token.expand(b, n, -1) + m["enc"].pos_embed,
                              tok)
            pred = m["reg"](m["enc"].forward_tokens(tok)).float()
            w = mask.float()[..., None]
            return ((pred - tgt.float()) ** 2 * w).sum() / (w.sum() * DIM)
        return model, loss, [("enc", "teacher")]

    raise ValueError(objective)


# ---------------------------------------------------------------------------
# Training loop (resumable)
# ---------------------------------------------------------------------------

def run_name(objective: str, seed: int) -> str:
    return f"{objective}_s{seed}"


def train(objective: str, seed: int, data_dir: str, out_dir: str, epochs: int = 10,
          batch: int = 512, lr: float = 1e-3, wd: float = 0.05, warmup_epochs: float = 1.0,
          ema: tuple = (0.996, 1.0), max_steps: int | None = None, ckpt_every: int = 500) -> str:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = load_data(*prepare_data(data_dir))
    n_img = len(y)
    steps_per_epoch = n_img // batch
    total = max_steps or epochs * steps_per_epoch
    warm = int(warmup_epochs * steps_per_epoch) if not max_steps else max(1, total // 10)

    model, loss_fn, pairs = build(objective)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    os.makedirs(out_dir, exist_ok=True)
    name = run_name(objective, seed)
    ck = os.path.join(out_dir, f"{name}.pt")
    step = 0
    if os.path.exists(ck):
        state = torch.load(ck, map_location=dev)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        print(f"[train] {name}: resumed at step {step}/{total}", flush=True)
    if step >= total:
        return ck

    batches = iter(BufferedBatches(x, y, batch, seed + 10_000 * step))
    t0, seen = time.time(), 0
    model.train()
    while step < total:
        xn, yn = next(batches)
        xb = torch.from_numpy(xn).pin_memory().to(dev, non_blocking=True) if dev == "cuda" \
            else torch.from_numpy(xn)
        yb = torch.from_numpy(yn).to(dev, non_blocking=True)
        frac = step / total
        cur = lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = cur
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            loss = loss_fn(model, gpu_augment(xb), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 3.0)
        scaler.step(opt)
        scaler.update()
        if pairs:                                                    # EMA teacher/target
            mom = ema[0] + (ema[1] - ema[0]) * frac
            with torch.no_grad():
                for a, b in pairs:
                    for ps, pt in zip(model[a].parameters(), model[b].parameters()):
                        pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
        step += 1
        seen += batch
        if step % 100 == 0:                                          # sync only every 100 steps
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"{name}: non-finite loss at step {step}")
            rate = seen / (time.time() - t0)
            eta = (total - step) * batch / max(rate, 1) / 60
            print(f"[train] {name} step {step}/{total} loss {loss.item():.4f} "
                  f"{rate:.0f} img/s eta {eta:.0f} min", flush=True)
        if step % ckpt_every == 0 or step == total:
            tmp = ck + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "objective": objective,
                        "seed": seed, "total": total}, tmp)
            os.replace(tmp, ck)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump({"objective": objective, "seed": seed, "steps": total, "batch": batch,
                   "epochs": epochs, "images": n_img}, fh)
    return ck


def load_encoder(ckpt: str, device: str = "cpu"):
    """The trained encoder alone (for feature extraction)."""
    import torch
    state = torch.load(ckpt, map_location="cpu")
    enc = build_encoder()
    sd = {k[4:]: v for k, v in state["model"].items() if k.startswith("enc.")}
    enc.load_state_dict(sd)
    return enc.eval().requires_grad_(False).to(device)


def knn_accuracy(enc, data_dir: str, device: str, n_train: int = 20000, n_val: int = 10000,
                 k: int = 20) -> float:
    """Sanity check of representation quality: k-NN top-1 on ImageNet-64 validation."""
    import torch
    import torch.nn.functional as F
    xt, yt = load_data(*prepare_data(data_dir, "train"))
    xv, yv = load_data(*prepare_data(data_dir, "validation"))
    rng = np.random.default_rng(0)
    it = np.sort(rng.choice(len(yt), n_train, replace=False))
    iv = np.sort(rng.choice(len(yv), min(n_val, len(yv)), replace=False))

    def feats(x, idx):
        out = []
        m = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        s = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        for i in range(0, len(idx), 1000):
            xb = torch.from_numpy(np.ascontiguousarray(x[idx[i:i + 1000]])).to(device)
            xb = (xb.permute(0, 3, 1, 2).float() / 255 - m) / s
            with torch.no_grad():
                out.append(F.normalize(enc(xb).float(), dim=1))
        return torch.cat(out)
    ft, fv = feats(xt, it), feats(xv, iv)
    lt = torch.from_numpy(yt[it].astype(np.int64)).to(device)
    lv = torch.from_numpy(yv[iv].astype(np.int64)).to(device)
    sim = fv @ ft.T
    top = sim.topk(k, dim=1).indices
    votes = torch.zeros(len(iv), 1000, device=device)
    votes.scatter_add_(1, lt[top], torch.ones_like(top, dtype=torch.float))
    return float((votes.argmax(1) == lv).float().mean())
