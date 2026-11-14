"""Regression check: our loader's smooth50 baseline vs the published baseline_smooth50.csv."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from possession import io  # noqa: E402

a = pd.read_csv(sys.argv[1]).label.to_numpy()
b = pd.read_csv(os.path.join(ROOT, "evaluation/results/baseline_smooth50.csv")).label.to_numpy()
g = io.graded_mask(len(a))
print("differing frames: all", int((a != b).sum()), "graded", int((a[g] != b[g]).sum()))
idx = (a != b).nonzero()[0]
print("first diffs", idx[:10].tolist(), list(zip(a[idx[:10]], b[idx[:10]])))
