<div align="center">

# The Heraclitean Representation Hypothesis

**What a network is trained to predict, not its size, sets which moment of perception it matches.**

[Paper (PDF)](paper/main.pdf) · [Methodology](METHODS.md) · [Run it](RUNNING.md) · [Cite](#citation)

![tests](https://github.com/LegallyCoder/hrh/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-2a78d6)
![license](https://img.shields.io/badge/license-MIT-1baf7a)

</div>

> *"You could not step twice into the same river."* (Heraclitus)

The [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) (Huh et al., 2024)
holds that as networks scale, their representations converge to one static model of reality.
The brain does not compute one representation, though: within ~400 ms, visual cortex moves
through retinotopic, semantic and recurrent regimes.

**HRH** treats a representation as a trajectory and asks which part of the brain's trajectory a
network can match. Our answer has three parts.

1. **Scale does not move it.** Across 25 models in six scaling ladders (6M to 1.8B parameters),
   alignment with THINGS-EEG2 is flat in model size in every time window. The same pipeline
   detects objective effects four to six times larger, so this is not a sensitivity problem.
2. **The objective fixes a spectral trajectory.** In deep linear networks under natural-image
   statistics, the layers of a trained network pass through a sequence of representational
   spectra that runs from the input's to an endpoint set by the objective. Width does not
   enter, and depth only refines the steps. Alignment with a brain stage peaks when the spectra
   match.
3. **A brain-free law, tested out of sample.** From properties measured without any neural data
   (spectral exponents, low-level and semantic retention), we fit a law on 30 development models
   and **pre-register its predictions for 18 held-out models** before their brain alignment is
   computed.

<p align="center"><img src="docs/teaser.png" width="92%" alt="Platonic vs Heraclitean convergence"></p>

## Results at a glance

| | |
|---|---|
| Scale | within six ladders (6M–1.8B params), alignment is flat in size in every window; model family explains 71–81% of the variance between models, size within family 0.1–2.2% |
| Theory | in deep linear networks, the objective fixes the path of layer spectra from the input's to an objective-specific endpoint; width does not enter |
| Held-out law (registered) | **supported**: the brain-free law ranks 18 unseen models at Spearman ρ = 0.65 / 0.70 / 0.63 (early / semantic / late); model size at chance |
| Convergence between models | agreement is highest in early layers and falls with depth; same-objective networks re-converge at the readout |

Pre-specified predictions that failed are reported in the paper.

## Quickstart

```bash
git clone https://github.com/LegallyCoder/hrh && cd hrh
pip install -e ".[dev]"          # numpy + matplotlib; add ".[models]" for torch/timm/transformers
make test                        # synthetic ground truth, theory, proxies, law (~20 s, CPU)
```

The full pipeline (GPU) is in [RUNNING.md](RUNNING.md) and `notebooks/hrh_colab.ipynb`.

## Layout

```
hrh/            align, brain, features, trajectory, theory, proxies, law, convergence,
                evaluate, plotting, paperfigs, paper_v2, data
run_hrh.py      brain | features | eval | proxies | law | predict | convergence | paper
registration/   registered held-out predictions (JSON + SHA-256)
paper/          LaTeX source; figures/ and generated/ are written by `run_hrh.py paper`
tests/          synthetic ground truth and theory checks (run in CI)
notebooks/      end-to-end GPU notebook
```

## Citation

```bibtex
@article{akkus2026hrh,
  title   = {The Heraclitean Representation Hypothesis: What a Network Learns to Predict,
             not its Size, Sets which Moment of Perception it Matches},
  author  = {Akku\c{s}, Talha R\"uzgar},
  year    = {2026},
  url     = {https://github.com/LegallyCoder/hrh}
}
```

## License

Code: MIT. THINGS-EEG2 and model weights are subject to their own licenses.
