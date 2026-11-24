# Registration

`predictions_heldout.json` holds the frozen law, the brain-free proxies of the held-out models and
the predicted window scores. `predictions_heldout.sha256` is its hash. Both are written by
`run_hrh.py predict`, registered externally, and committed here **before** any held-out brain
analysis. `hrh/evaluate.py` refuses to analyse held-out models unless the JSON matches the hash.
