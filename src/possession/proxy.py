"""Proxy benchmark on OTHER games: Metrica sample games 1 and 2.

The ranking metric cannot be measured on BAR-ATH (the public window is 17 possessions, noise band
+/-0.33, and tuning on it is forbidden). The sample games have human event annotation, from which
calibrate.reference_labels builds a per-frame H/A/D reference by a written-down rule. We run the
SAME pipeline on their tracking and score it with the official evaluation/possession_metrics.py.
Design choices are made on this benchmark, never on BAR-ATH labels.

Two caveats, stated in the technical document:
  * the reference is event-derived, not corrected-tracking-derived like the challenge GT, so it
    has its own conventions (e.g. dead-ball boundaries at event timestamps);
  * the sample tracking is cleaner. `--degrade` removes ball observations in contiguous gaps
    (seeded, deterministic) until the alive-frame missing share matches BAR-ATH's 40.7 %.

    python -m possession.proxy [--degrade]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

from . import ROOT
from . import calibrate, features, run

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

PITCH = np.array(calibrate.PITCH, dtype=np.float32)
BARATH_MISSING_ALIVE = 0.407        # DATA.md: ball missing on 41% of graded frames (64,643/158,857)


def load_game(game):
    frames, period, home, ball = calibrate._tracking(game, "Home")
    _, _, away, _ = calibrate._tracking(game, "Away")
    players = np.concatenate([home, away], axis=1).astype(np.float32)
    team = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
    speed = np.full(players.shape[:2], np.nan, dtype=np.float32)
    speed[1:] = np.linalg.norm(np.diff(players, axis=0), axis=2) * 25
    events = pd.read_csv(calibrate._fetch(game, "RawEventsData"))
    ref = calibrate.reference_labels(events, len(frames))
    blocks = []
    for p in (1, 2):
        idx = np.flatnonzero((period == p) & (ref >= 0))
        if len(idx):
            blocks.append((p, int(idx[0]), int(idx[-1])))
    atd = {"ball": ball.astype(np.float32), "players": players, "speed": speed, "team": team,
           "pitch": PITCH}
    return atd, ref, blocks


def degrade(atd, ref, rng):
    """Drop ball observations in gaps with BAR-ATH-like lengths until the alive missing share
    reaches BARATH_MISSING_ALIVE. Gaps start preferentially near players (duels), where the real
    tracker loses the ball: start frames are sampled with weight 1/(1 + nearest distance)."""
    ball = atd["ball"].copy()
    alive = (ref == 0) | (ref == 1)
    d = np.nanmin(np.linalg.norm(atd["players"] - ball[:, None], axis=2), axis=1)
    w = np.where(alive & np.isfinite(d), 1.0 / (1.0 + np.nan_to_num(d, nan=50)), 0)
    w /= w.sum()
    while np.mean(~np.isfinite(ball[alive, 0])) < BARATH_MISSING_ALIVE:
        s = rng.choice(len(ball), p=w)
        L = int(np.clip(rng.lognormal(np.log(25), 1.0), 3, 750))   # median 1 s, long tail
        ball[s:s + L] = np.nan
    return dict(atd, ball=ball)


def score(labels, ref, blocks):
    gt = np.full(len(ref), 3, dtype=np.int8)                         # X
    for _, a, b in blocks:
        gt[a:b + 1] = np.where(ref[a:b + 1] == 2, 2, ref[a:b + 1])
    pred = pm.label_codes(labels)
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--config", default=run.CONFIG)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    rows = []
    for game in calibrate.GAMES:
        atd, ref, blocks = load_game(game)
        if args.degrade:
            atd = degrade(atd, ref, np.random.default_rng(game))
        F = features.build(atd)
        if args.baseline:
            labels = run.baseline_smooth(atd, F)
            mask = np.zeros(len(labels), bool)
            for _, a, b in blocks:
                mask[a:b + 1] = True
            labels = np.where(mask, labels, "X")
        else:
            labels, info = run.predict(atd, F, dict(cfg), blocks=blocks, smart_tagging=None)
        r = score(labels, ref, blocks)
        rows.append((game, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
        print("game {}  spellF1 {:.4f}  macroF1 {:.4f}  poss {} / gt {}".format(*rows[-1]),
              " aliveDeadF1 {:.3f} teamAcc {:.3f} D P/R {:.3f}/{:.3f}".format(
                  r["alive_dead_f1"], r["team_acc_both_alive"], r["per_class"]["D"]["precision"],
                  r["per_class"]["D"]["recall"]))
    print("MEAN  spellF1 {:.4f}  macroF1 {:.4f}".format(np.mean([r[1] for r in rows]),
                                                        np.mean([r[2] for r in rows])))


if __name__ == "__main__":
    main()
