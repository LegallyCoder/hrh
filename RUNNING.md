# Running the pipeline

One GPU (A100 or L4 recommended; a T4 works) and about 30 GB of temporary disk.
Everything is written to `$OUT` as it is produced, so an interrupted step resumes where it
stopped. `notebooks/hrh_colab.ipynb` runs every step below.

| step | command | what it does |
|---|---|---|
| install | `pip install -e ".[models,dev]" && pytest -q` | dependencies and tests |
| brain | `python run_hrh.py brain --out $OUT` | THINGS-EEG2 download (10 × 1.07 GB), cross-validated RDMs, noise ceilings |
| features | `python run_hrh.py features --out $OUT --clean-cache` | 51 networks, every block, 3 poolings; only Gram matrices are kept |
| development eval | `python run_hrh.py eval --out $OUT --paper $OUT/paper` | scaling and objective analyses; held-out models excluded |
| proxies | `python run_hrh.py proxies --out $OUT` | brain-free proxies for every model (CPU) |
| law | `python run_hrh.py law --out $OUT` | fit and freeze the law on the development models (CPU) |
| predict | `python run_hrh.py predict --out $OUT --reg-dir registration` | held-out predictions and SHA-256 |
| register | OSF, then commit `registration/` | must happen before the next step |
| confirmatory eval | `python run_hrh.py eval --out $OUT --paper $OUT/paper` | adds `law_test` with its verdict |

## Output layout (`$OUT`)

| path | contents |
|---|---|
| `brain/brain_sub-XX.npz` | per-subject RDMs and ceilings |
| `grams/<model>.npz` | per-layer Gram matrices |
| `summary_{grid4,gap,cls}.json` | slopes, contrasts, verdicts, `law_test` |
| `models_*.csv` | per-model numbers |
| `proxies_grid4.json`, `law.json` | brain-free proxies and the frozen law |
