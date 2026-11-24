# Methodology

## 1. Questions

**PRH** (Huh et al., 2024): as networks scale, representations converge to one static kernel,
so alignment with any competent system, the brain included, should rise with scale.

**HRH**: a representation is a trajectory over computation time. Which part of the brain's
trajectory a network can match is set by its training objective, through the sequence of
representational spectra its layers pass through, and not by its size.

## 2. Theory (`hrh/theory.py`, checked in `tests/test_theory.py`)

Deep linear networks with weight decay, population loss.

| Result | Statement |
|---|---|
| P1 nuisance erasure | at every stationary point, no layer carries directions outside the task subspace |
| P2 power collapse | layers are balanced and G_t = G_T^{t/T}: the whole trajectory is fixed by the end-to-end map |
| T1 trajectory ceiling | alignment of any layer with an off-task stage is bounded by ρ_S, independently of width, depth and λ |
| P4 spectral matching | with input covariance Σ, CKA between whitened Grams Σ^a and Σ^b is unimodal in a and peaks at a = b |
| P5 spectral trajectory | for a target M = Σ^c, layer t has whitened Gram ∝ Σ^{1+2ct/T}: exponents run linearly from the input's (1) to the objective's endpoint. Pixel reconstruction stays at 1; a whitened latent target sweeps [0, 1]. Width does not enter; depth refines the grid. |

P4 and P5 together predict that (i) scale leaves the alignment profile unchanged, and (ii) the
objective decides which brain stages are reachable.

## 3. Development study (33 models, THINGS-EEG2, 10 subjects)

* Brain: cross-validated RDMs per 10 ms bin, split-half noise ceilings, reliable bins tested
  against the pre-stimulus baseline (`hrh/brain.py`, `trajectory.reliable_mask`).
* Models: every block hooked; grid4 (primary), gap and cls pooling (`hrh/features.py`).
* Window scores (early 50–120, semantic 120–250, late 250–600 ms): best layer selected on one
  half of the repetitions, scored on the other (`trajectory.window_scores`).
* Scaling: within-family slope on log10(params) with family fixed effects; hierarchical
  bootstrap over subjects × stimuli (`trajectory.bootstrap_window_stats`).

Results used here. All scaling slopes are ≈0, with every 95% interval within ±0.014 per decade.
At matched size, objectives differ by up to ±0.09 (I-JEPA above, MAE below its references).
The intervals in this study treat the models as fixed. The held-out test below also resamples
models.

## 4. Brain-free proxies (`hrh/proxies.py`)

Computed from Gram matrices over the 200 test images, the images themselves and human category
labels only. No neural data enter.

| Proxy | Definition |
|---|---|
| α(ℓ) | power-law exponent of the eigenvalues of the double-centred Gram, ranks 2–50 |
| L(ℓ) | RSA with a Gabor-energy model: 4 wavelengths × 8 orientations, 128 px, 4×4 pooling |
| S(ℓ) | RSA with Jaccard distances between THINGSplus category sets (53 categories; 194/200 images labelled) |

## 5. The law (`hrh/law.py`)

For window w:

    score_w = c0 + c1·L_max + c2·S_max − c3·m_w²,    m_w = min_ℓ |α(ℓ) − b_w|

b_w is chosen on a fixed grid (0 to 3, step 0.05) by leave-one-family-out cross-validation, and
c by least squares, on the 33 development models. The fitted law is then frozen. Baselines are
log10(params) and a constant.

## 6. Held-out confirmatory test

18 models never analysed before, from seven objectives: DINO, ConvNeXt V2-FCMAE (pixel
reconstruction in a CNN), EVA-02 (masked CLIP-feature prediction), SigLIP, AIMv2, data2vec and
BEiT (`features.HELDOUT_LADDERS`).

Order, enforced in code:

1. Extract Grams (no brain data).
2. `run_hrh.py proxies`, `law`, `predict`. These write `registration/predictions_heldout.json`
   and its SHA-256.
3. Register the predictions externally (OSF) and commit both files.
4. Only then does `evaluate` include held-out models. It refuses if the JSON does not match the
   hash.

**Decision rule** (`law.law_verdict`): Spearman ρ between predicted and observed held-out window
scores, with a bootstrap over subjects, stimuli **and models**.

* **supported**: lower 95% bound of ρ > 0 in the early and late windows, and the law's ρ exceeds
  the log-params baseline's in both.
* **partial**: exactly one window passes.
* otherwise **not supported**.

The scale null is replicated on the held-out ladders (`slope_heldout`).

## 7. Controlled objective experiment (pre-specified before any training)

The off-the-shelf comparison cannot separate the objective from data, architecture and recipe.
Here only the objective varies (`hrh/train.py`). One architecture (ViT-Tiny, dim 192, depth 12,
8×8 patches on 64×64 images, no class token), one dataset (ImageNet-1k at 64×64), identical
augmentation, optimizer, schedule and number of images seen (10 epochs). Four objectives:
supervised classification, MAE (masked pixels), JEPA (EMA-target embeddings of masked blocks),
data2vec (EMA teacher's averaged top-layer features at masked positions). Two seeds each.
THINGS images are presented at 64 px.

Predictions (`evaluate.controlled_verdict`; window scores averaged over seeds; bootstrap over
subjects and stimuli):

* **H1 (primary)**: latent prediction (JEPA and data2vec together) exceeds **both** supervised and
  MAE networks in the early window: lower 95% bounds of both contrasts > 0.
* **H2**: the same in the late window.
* **H3 (theory)**: in each seed, the MAE network's final-layer spectral exponent is higher than
  those of both latent-prediction networks.

Sanity check, reported for every run: k-NN top-1 accuracy on ImageNet-64 validation.

## 8. Controls

| Concern | Control |
|---|---|
| Late EEG is noise | per-bin ceilings; unreliable bins excluded |
| Metric artefact | RSA, debiased CKA, mutual k-NN |
| Layer chosen on test data | split-half selection |
| Pooling discards retinotopy | grid4, gap and cls reported |
| Scale confounded with objective | within-family slopes |
| Proxy tuned on the test models | proxies and law frozen before held-out data; hash-checked |
| Few held-out models | models resampled in the bootstrap |
| Pipeline bug | synthetic ground truth in both directions, run in CI |
