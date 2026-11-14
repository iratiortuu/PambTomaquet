# PambTomaquet: More than a Hack 2027, Phase 1

Ball possession (H / A / D) for every frame of BAR-ATH, from the tactical tracking and Smart
Tagging. A three-state HMM is decoded with Viterbi, and its parameters are calibrated on Metrica's
public sample games. The method is in [PambTomaquet.md](PambTomaquet.md).

`games.py`, `evaluation/` and `starter/` are copied unchanged from the official challenge
repository, [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(MIT licence, see `LICENSE`). Our code is in `src/possession/`, `config/`, `scripts/`, `tests/`
and `notebooks/`. The submitted file is in `submission/PambTomaquet.csv`.

## Run

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-pambtomaquet.txt
# match data: unpack into ./data (layout in data/README.md) or export MTAH_DATA_DIR=/path
PYTHONPATH=src .venv/bin/python -m possession.run
# -> evaluation/results/PambTomaquet.csv, validated with evaluation/validate_submission.py
```

| command | what |
|---|---|
| `python -m possession.calibrate` | download sample games 1–2 into `external/` and rebuild `config/calibration.yaml` |
| `python -m possession.run --baseline` | the published smooth50 baseline on our loader (regression check) |
| `python scripts/sensitivity.py` | ±20% parameter sweep, no ground truth used |
| `pytest -q tests` | unit tests plus a two-run determinism check |

No match data, videos, caches or downloaded sample data are committed (`.gitignore`: `data/*`,
`cache/`, `external/`, `*.npz`, `*.mp4`).
