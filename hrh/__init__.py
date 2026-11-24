"""Heraclitean Representation Hypothesis.

Modules
-------
align       kernel/RDM alignment metrics (RSA, debiased CKA, mutual k-NN)
brain       THINGS-EEG2 loading, cross-validated RDMs per time bin, noise ceilings
features    multi-layer frozen feature extraction -> per-layer Gram matrices (torch)
trajectory  layer x time alignment, coverage, DTW, window scores, hierarchical bootstrap
theory      deep linear networks: erasure, power collapse, ceiling, spectral matching
proxies     brain-free model properties (spectral exponent, low-level/semantic retention)
law         the matched-spectrum law: fit, freeze, predict, registered verdict
evaluate    full protocol -> results, registered tests, paper assets
"""
