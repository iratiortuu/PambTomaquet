"""Unsupervised stability check: perturb each key parameter by +/-20 % and report how the decoded
possession structure moves, and how many frames change label vs the frozen config.
No ground truth is read."""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from possession import features, io, run  # noqa: E402

KEYS = ["obs_radius_m", "control_radius_m", "p_nearest_correct", "mean_possession_s",
        "mean_dead_s", "dead_share_of_possession_ends", "restart_ball_speed", "duel_margin_m"]

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "model.yaml")))
atd = io.load_atd()
F = features.build(atd)
g = io.graded_mask(len(F["ball_seen"]))
base, info = run.predict(atd, F, dict(cfg))
cfg["temperature"] = info["temperature"]          # hold the temperature fixed in the sweep
s0 = run.summary(base)
print("frozen: possessions {possessions}  mean {mean_possession_s}s  dead {dead_balls}".format(**s0))
print("{:<32}{:>6}{:>8}{:>8}{:>7}{:>10}".format("parameter", "x", "poss", "mean_s", "dead", "changed%"))
for k in KEYS:
    for f in (0.8, 1.2):
        c = dict(cfg)
        c[k] = min(c[k] * f, 0.99) if k == "p_nearest_correct" else c[k] * f
        lab, _ = run.predict(atd, F, c)
        s = run.summary(lab)
        print("{:<32}{:>6}{:>8}{:>8}{:>7}{:>10.2f}".format(
            k, f, s["possessions"], s["mean_possession_s"], s["dead_balls"],
            100 * np.mean(lab[g] != base[g])))
