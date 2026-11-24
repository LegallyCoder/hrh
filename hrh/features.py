"""Multi-layer frozen features -> per-layer Gram matrices.

Every residual / attention block is hooked and every model gets the same
poolings, so models are compared as representational hierarchies rather than
through heterogeneous output heads (logits, projections, class tokens).

Pooling is part of the result. Global average pooling discards retinotopy,
which is what early EEG carries, so three poolings are stored and reported:
  gap    global average over space (and time, for video models)
  grid4  4x4 spatial average (coarse retinotopy; pre-registered primary)
  cls    class token (ViTs that have one)

Only N x N Gram matrices are written: a few MB per model even at 1B+ params,
and every metric in align.py is computed from them.

Two model sets
--------------
LADDERS            the pre-registered scaling ladders: only size varies within a
                   family. The primary scaling test uses these and nothing else.
OBJECTIVE_LADDERS  dense / predictive self-supervised objectives (MAE, I-JEPA,
                   V-JEPA 2) for the objective test (Proposition 3), compared with
                   size-matched single-readout references (REFERENCE_BY_SIZE).
"""
from __future__ import annotations

import os

import numpy as np

# family -> [(model_name, params_M)]. Names are timm pretrained tags except
# 'tv:' = torchvision. Verify with timm.list_pretrained() if a tag 404s;
# the runner fails soft per model.
LADDERS: dict[str, list[tuple[str, float]]] = {
    "classic-cnn":    [("tv:alexnet", 61.1), ("vgg16.tv_in1k", 138.4)],
    "resnet-in1k":    [("resnet18.tv_in1k", 11.7), ("resnet34.tv_in1k", 21.8),
                       ("resnet50.tv_in1k", 25.6), ("resnet101.tv_in1k", 44.5),
                       ("resnet152.tv_in1k", 60.2)],
    "convnext-in22k": [("convnext_tiny.fb_in22k_ft_in1k", 28.6),
                       ("convnext_small.fb_in22k_ft_in1k", 50.2),
                       ("convnext_base.fb_in22k_ft_in1k", 88.6),
                       ("convnext_large.fb_in22k_ft_in1k", 197.8),
                       ("convnext_xlarge.fb_in22k_ft_in1k", 350.2)],
    "vit-sup":        [("vit_tiny_patch16_224.augreg_in21k_ft_in1k", 5.7),
                       ("vit_small_patch16_224.augreg_in21k_ft_in1k", 22.1),
                       ("vit_base_patch16_224.augreg_in21k_ft_in1k", 86.6),
                       ("vit_large_patch16_224.augreg_in21k_ft_in1k", 304.3)],
    "dinov2":         [("vit_small_patch14_dinov2.lvd142m", 22.1),
                       ("vit_base_patch14_dinov2.lvd142m", 86.6),
                       ("vit_large_patch14_dinov2.lvd142m", 304.4),
                       ("vit_giant_patch14_dinov2.lvd142m", 1136.5)],
    "clip-laion":     [("vit_base_patch16_clip_224.laion2b", 86.2),
                       ("vit_large_patch14_clip_224.laion2b", 304.0),
                       ("vit_huge_patch14_clip_224.laion2b", 632.1),
                       ("vit_giant_patch14_clip_224.laion2b", 1012.6),
                       ("vit_gigantic_patch14_clip_224.laion2b", 1844.9)],
}

# Objective test (registered before these models were run). 'hf:' = Hugging Face
# transformers checkpoint. Parameter counts are encoder-only.
OBJECTIVE_LADDERS: dict[str, list[tuple[str, float]]] = {
    "mae":    [("vit_base_patch16_224.mae", 85.8), ("vit_large_patch16_224.mae", 303.3),
               ("vit_huge_patch14_224.mae", 630.8)],
    "ijepa":  [("hf:facebook/ijepa_vith14_22k", 630.8),
               ("hf:facebook/ijepa_vitg16_22k", 1011.0)],
    "vjepa2": [("hf:facebook/vjepa2-vitl-fpc64-256", 303.9),
               ("hf:facebook/vjepa2-vith-fpc64-256", 631.8),
               ("hf:facebook/vjepa2-vitg-fpc64-256", 1012.0)],
}
SIZE_CLASS = {   # architecture size class of every model that enters the objective test
    "vit_base_patch16_224.mae": "B", "vit_large_patch16_224.mae": "L",
    "vit_huge_patch14_224.mae": "H", "hf:facebook/ijepa_vith14_22k": "H",
    "hf:facebook/ijepa_vitg16_22k": "g", "hf:facebook/vjepa2-vitl-fpc64-256": "L",
    "hf:facebook/vjepa2-vith-fpc64-256": "H", "hf:facebook/vjepa2-vitg-fpc64-256": "g",
}
# single global readout (supervised classification, image-text contrastive)
REFERENCE_BY_SIZE = {
    "B": ["vit_base_patch16_224.augreg_in21k_ft_in1k", "vit_base_patch16_clip_224.laion2b"],
    "L": ["vit_large_patch16_224.augreg_in21k_ft_in1k", "vit_large_patch14_clip_224.laion2b"],
    "H": ["vit_huge_patch14_clip_224.laion2b"],
    "g": ["vit_giant_patch14_clip_224.laion2b"],
}
VIDEO_FRAMES = 16          # V-JEPA 2 sees a still image as a static 16-frame clip

# Held-out set for the law test (hrh/law.py). Chosen before any of these models
# was run; their brain alignment is not computed until the law's predictions for
# them are registered (registration/predictions_heldout.json + .sha256).
# Parameter counts: vision encoder only, approximate.
HELDOUT_LADDERS: dict[str, list[tuple[str, float]]] = {
    "dino":        [("vit_small_patch16_224.dino", 21.7), ("vit_base_patch16_224.dino", 85.8)],
    "fcmae":       [("convnextv2_atto.fcmae", 3.4), ("convnextv2_tiny.fcmae", 27.9),
                    ("convnextv2_base.fcmae", 87.7), ("convnextv2_large.fcmae", 196.4),
                    ("convnextv2_huge.fcmae", 657.5)],
    "eva02-mim":   [("eva02_base_patch14_224.mim_in22k", 85.8),
                    ("eva02_large_patch14_224.mim_in22k", 303.3)],
    "siglip":      [("vit_base_patch16_siglip_224.webli", 92.9),
                    ("vit_large_patch16_siglip_256.webli", 315.9),
                    ("vit_so400m_patch14_siglip_224.webli", 427.7)],
    "aimv2":       [("aimv2_large_patch14_224.apple_pt", 309.2),
                    ("aimv2_huge_patch14_224.apple_pt", 681.0)],
    "data2vec":    [("hf:facebook/data2vec-vision-base", 85.8),
                    ("hf:facebook/data2vec-vision-large", 303.3)],
    "beit-pt22k":  [("hf:microsoft/beit-base-patch16-224-pt22k", 85.8),
                    ("hf:microsoft/beit-large-patch16-224-pt22k", 303.3)],
}

# Controlled objective experiment (hrh/train.py): identical ViT-Tiny/8 at 64 px, identical
# data and budget; only the objective differs. 'local:' = checkpoint in --ckpt-dir.
CONTROLLED_LADDERS: dict[str, list[tuple[str, float]]] = {
    f"ctl-{o}": [(f"local:{o}_s{s}", 5.4) for s in (0, 1)]
    for o in ("sup", "mae", "jepa", "data2vec")}
CONTROLLED = {m for fam in CONTROLLED_LADDERS.values() for m, _ in fam}
CKPT_DIR = os.environ.get("HRH_CKPT_DIR", "checkpoints")

ALL_LADDERS = {**LADDERS, **OBJECTIVE_LADDERS, **HELDOUT_LADDERS, **CONTROLLED_LADDERS}
PARAMS = {m: p for fam in ALL_LADDERS.values() for m, p in fam}
FAMILY = {m: f for f, fam in ALL_LADDERS.items() for m, _ in fam}
PRIMARY = {m for fam in LADDERS.values() for m, _ in fam}
HELDOUT = {m for fam in HELDOUT_LADDERS.values() for m, _ in fam}
DEVELOPMENT = PRIMARY | {m for fam in OBJECTIVE_LADDERS.values() for m, _ in fam}

BLOCK_CLASSES = {"BasicBlock", "Bottleneck", "ConvNeXtBlock", "Block",
                 "ResidualAttentionBlock", "NestedTensorBlock"}
POOLS = ("gap", "grid4", "cls")


IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _image_tf(size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    from torchvision import transforms as T
    return T.Compose([T.Resize(int(round(size / 0.875)), interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(size), T.ToTensor(), T.Normalize(tuple(mean), tuple(std))])


def _hf_norm(repo: str):
    """Normalization constants from the checkpoint's own preprocessor (BEiT and
    data2vec use 0.5/0.5); ImageNet constants if it has none."""
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(repo)
        mean, std = getattr(proc, "image_mean", None), getattr(proc, "image_std", None)
        if mean is not None and std is not None:
            return mean, std
    except Exception:                                # video-only or missing processor
        pass
    return IMAGENET_MEAN, IMAGENET_STD


def _load(name: str, device: str):
    import torch
    if name.startswith("local:"):                   # controlled experiment, 64 px input
        from torchvision import transforms as T

        from .train import IMG, MEAN, STD, load_encoder
        net = load_encoder(os.path.join(CKPT_DIR, name[6:] + ".pt"), device)
        tf = T.Compose([T.Resize(IMG, interpolation=T.InterpolationMode.BICUBIC),
                        T.CenterCrop(IMG), T.ToTensor(), T.Normalize(MEAN, STD)])
        return net, tf, [(f"blocks.{i}", b) for i, b in enumerate(net.blocks)], 0
    if name.startswith("hf:"):                      # transformers: hidden states, no hooks
        from transformers import AutoConfig, AutoModel
        repo = name[3:]
        cfg = AutoConfig.from_pretrained(repo)
        try:                                        # fused attention: O(tokens) memory
            net = AutoModel.from_pretrained(repo, attn_implementation="sdpa")
        except (ValueError, TypeError):
            net = AutoModel.from_pretrained(repo)
        size = int(getattr(cfg, "crop_size", None) or cfg.image_size)
        net.eval().requires_grad_(False).to(device)
        return net, _image_tf(size, *_hf_norm(repo)), [], 0
    if name.startswith("tv:"):
        from torchvision import models as tvm, transforms as T
        net = getattr(tvm, name[3:])(weights="IMAGENET1K_V1")
        tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                        T.CenterCrop(224), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        hook_mods = [(n, m) for n, m in net.named_modules()
                     if isinstance(m, torch.nn.ReLU)]
        prefix = 0
    else:
        import timm
        net = timm.create_model(name, pretrained=True)
        cfg = timm.data.resolve_data_config({}, model=net)
        tf = timm.data.create_transform(**cfg)
        hook_mods = [(n, m) for n, m in net.named_modules()
                     if type(m).__name__ in BLOCK_CLASSES]
        if not hook_mods and isinstance(getattr(net, "blocks", None), torch.nn.Module):
            hook_mods = [(f"blocks.{i}", m) for i, m in enumerate(net.blocks)]  # EVA, AIMv2, ...
        if not hook_mods:                           # plain stacks (VGG): every ReLU
            hook_mods = [(n, m) for n, m in net.named_modules()
                         if isinstance(m, torch.nn.ReLU)]
        if not hook_mods:
            raise RuntimeError(f"{name}: no layers to hook")
        prefix = int(getattr(net, "num_prefix_tokens", 0))
    for _n, m in hook_mods:            # torchvision ReLUs are inplace: copy-safe hook
        if hasattr(m, "inplace"):
            m.inplace = False
    net.eval().requires_grad_(False).to(device)
    return net, tf, hook_mods, prefix


def _pool(out, prefix: int, spatial: int | None = None) -> dict:
    """spatial: tokens per frame for video models (time groups are averaged;
    patch tokens are time-major after the 3D patch embedding)."""
    import torch.nn.functional as F
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = out.float()
    if spatial and out.ndim == 3 and out.shape[1] % spatial == 0 and out.shape[1] > spatial:
        out = out.reshape(out.shape[0], out.shape[1] // spatial, spatial, -1).mean(1)
    res = {}
    if out.ndim == 4:                                   # (B, C, H, W)
        res["gap"] = out.mean((2, 3))
        res["grid4"] = F.adaptive_avg_pool2d(out, 4).flatten(1)
    elif out.ndim == 3:                                 # (B, tokens, D)
        patches = out[:, prefix:]
        side = int(round(patches.shape[1] ** 0.5))
        res["gap"] = patches.mean(1)
        if side * side == patches.shape[1]:
            grid = patches.transpose(1, 2).reshape(out.shape[0], -1, side, side)
            res["grid4"] = F.adaptive_avg_pool2d(grid, 4).flatten(1)
        if prefix > 0:
            res["cls"] = out[:, 0]
    elif out.ndim == 2:
        res["gap"] = res["grid4"] = out
    return {k: v.cpu().numpy() for k, v in res.items()}


def _hf_pool(h, spatial: int, video: bool) -> dict:
    """Image models may prepend class/register tokens: everything beyond the
    patch grid is treated as prefix (and the first token as cls)."""
    if video:
        return _pool(h, 0, spatial)
    t = h[0].shape[1] if isinstance(h, (tuple, list)) else h.shape[1]
    return _pool(h, max(t - spatial, 0), None)


def _hf_forward(net, x, video: bool):
    """Per-layer hidden states and final encoder output of a transformers model.
    V-JEPA 2 gets a static clip; its predictor is skipped where the API allows,
    and encoder layers are hooked in case hidden states are not returned."""
    if not video:
        out = net(pixel_values=x, output_hidden_states=True)
        return list(out.hidden_states[1:]), out.last_hidden_state
    clip = x.unsqueeze(1).repeat(1, VIDEO_FRAMES, 1, 1, 1)          # (B, T, C, H, W)
    layers = getattr(getattr(net, "encoder", None), "layer", None)
    caught: list = []
    handles = [m.register_forward_hook(
        lambda _m, _i, o: caught.append(o[0] if isinstance(o, (tuple, list)) else o))
        for m in (layers or [])]
    try:
        out, err = None, None
        for kw in ({"skip_predictor": True}, {}):
            caught.clear()
            try:
                out = net(pixel_values_videos=clip, output_hidden_states=True, **kw)
                break
            except TypeError as exc:                 # older/newer forward signature
                err = exc
        if out is None:
            raise err
    finally:
        for h in handles:
            h.remove()
    hidden = getattr(out, "hidden_states", None)
    hidden = list(hidden[1:]) if hidden else caught
    if not hidden:
        raise RuntimeError("V-JEPA 2: no encoder hidden states returned or hooked")
    return hidden, out.last_hidden_state


def extract_grams(name: str, img_paths: list[str], device: str, out_dir: str,
                  batch: int | None = None) -> str:
    """Hook every block, pool, and save Gram matrices. Returns the .npz path.

    Half precision first; if the GPU has no kernel for a layer in half precision
    (cuDNN "unable to find an engine", seen for ConvNeXt V2 on T4), retry in full
    precision, and finally with cuDNN disabled."""
    import gc

    import torch
    dest = os.path.join(out_dir, f"{name.replace(':', '_').replace('/', '_')}.npz")
    if os.path.exists(dest):
        return dest
    attempts = [(True, True), (False, True), (False, False)]          # (amp, cudnn)
    for k, (amp, cudnn) in enumerate(attempts):
        torch.backends.cudnn.enabled = cudnn
        try:
            return _extract(name, img_paths, device, out_dir, batch, amp)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if k == len(attempts) - 1 or not ("engine" in msg or "cudnn" in msg):
                raise
            print(f"[features] {name}: retrying (amp={attempts[k + 1][0]}, "
                  f"cudnn={attempts[k + 1][1]}) after: {exc}", flush=True)
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        finally:
            torch.backends.cudnn.enabled = True
    return dest


def _extract(name: str, img_paths: list[str], device: str, out_dir: str,
             batch: int | None, amp: bool) -> str:
    import torch
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{name.replace(':', '_').replace('/', '_')}.npz")
    net, tf, hook_mods, prefix = _load(name, device)
    store: dict[str, dict[str, list]] = {n: {} for n, _ in hook_mods}
    handles = []
    for lname, mod in hook_mods:
        def hook(_m, _i, out, lname=lname):
            for pool, arr in _pool(out, prefix).items():
                store[lname].setdefault(pool, []).append(arr)
        handles.append(mod.register_forward_hook(hook))

    if batch is None:
        batch = 8 if PARAMS.get(name, 0) > 900 else 16 if PARAMS.get(name, 0) > 250 else 50
    use_amp = amp and device.startswith("cuda")
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16)
    hf = name.startswith("hf:")
    video = hf and getattr(net.config, "model_type", "") == "vjepa2"
    if video:            # a 16-frame clip is 2048 tokens per image: keep batches small
        batch = min(batch, 2)
    spatial = None
    if hf:
        size = int(getattr(net.config, "crop_size", None) or net.config.image_size)
        spatial = (size // net.config.patch_size) ** 2
    hf_layers: list[str] = []
    final = []
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        for i in range(0, len(img_paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB"))
                             for p in img_paths[i:i + batch]]).to(device)
            if hf:
                hs, last = _hf_forward(net, x, video)
                hf_layers = [f"layer{k:02d}" for k in range(len(hs))]
                for lname, h in zip(hf_layers, hs):
                    st = store.setdefault(lname, {})
                    for pool, arr in _hf_pool(h, spatial, video).items():
                        st.setdefault(pool, []).append(arr)
                emb = _hf_pool(last, spatial, video)["gap"]
                final.append(emb)
                continue
            if hasattr(net, "forward_head"):            # timm: pre-logit embedding
                emb = net.forward_head(net.forward_features(x), pre_logits=True)
            else:
                emb = net(x)
            final.append(emb.float().flatten(1).cpu().numpy())
    for h in handles:
        h.remove()

    layer_names = [n for n, _ in hook_mods] + hf_layers + ["__embed__"]
    store["__embed__"] = {"gap": final, "grid4": final}
    payload = {"layer_names": np.array(layer_names), "params_m": PARAMS.get(name, np.nan)}
    for pool in POOLS:
        grams, sums, dims, ok = [], [], [], []
        for ln in layer_names:
            if pool in store[ln]:
                f = np.concatenate(store[ln][pool]).astype(np.float64)
                grams.append(f @ f.T)
                sums.append(f.sum(1))
                dims.append(f.shape[1])
                ok.append(True)
            else:
                n = len(img_paths)
                grams.append(np.full((n, n), np.nan))
                sums.append(np.full(n, np.nan))
                dims.append(0)
                ok.append(False)
        payload[f"gram_{pool}"] = np.stack(grams).astype(np.float32)
        payload[f"sums_{pool}"] = np.stack(sums).astype(np.float32)
        payload[f"dims_{pool}"] = np.array(dims)
        payload[f"ok_{pool}"] = np.array(ok)
    np.savez_compressed(dest, **payload)
    del net
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return dest


def model_rdms(npz_path: str, pool: str = "grid4") -> tuple[np.ndarray, list[str], float]:
    """Load saved Grams -> (L, N, N) correlation-distance RDMs for valid layers."""
    from .align import gram_to_corr_rdm
    z = np.load(npz_path, allow_pickle=False)
    ok = z[f"ok_{pool}"]
    names = [str(n) for n, k in zip(z["layer_names"], ok) if k]
    if not names:                                   # e.g. cls pooling on a CNN
        return np.empty((0, 0, 0)), [], float(z["params_m"])
    g, s, d = z[f"gram_{pool}"][ok], z[f"sums_{pool}"][ok], z[f"dims_{pool}"][ok]
    rdms = np.stack([gram_to_corr_rdm(g[i].astype(np.float64), s[i].astype(np.float64),
                                      int(d[i])) for i in range(len(g))])
    return rdms, names, float(z["params_m"])
