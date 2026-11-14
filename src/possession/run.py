"""End-to-end: tactical ATD (+ Smart Tagging) -> PambTomaquet.csv.

    python -m possession.run [--out evaluation/results/PambTomaquet.csv] [--baseline]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import yaml

from . import ROOT
from . import deadball, features, hmm, io, tags
from games import half_video_blocks, GAME  # noqa: E402

LABELS = np.array(["H", "A", "D"])
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "PambTomaquet.csv")
CONFIG = os.path.join(ROOT, "config", "model.yaml")


def baseline_smooth(atd, F, window=50):
    """The published baseline (nearest player, majority vote), on our loader -- a regression test."""
    codes = np.where(~F["ball_seen"], -1, np.where(F["d_home"] <= F["d_away"], 0, 1))
    codes = np.where(np.isfinite(F["d_home"]) | np.isfinite(F["d_away"]), codes, -1)
    ch = np.r_[0, np.cumsum(codes == 0)]
    ca = np.r_[0, np.cumsum(codes == 1)]
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = ch[hi] - ch[lo], ca[hi] - ca[lo]
    return np.where(h + a == 0, "D", np.where(h >= a, "H", "A"))


def observation(F, cfg):
    dh = np.nan_to_num(F["d_home"], nan=np.inf)
    da = np.nan_to_num(F["d_away"], nan=np.inf)
    near = np.minimum(dh, da)
    obs = np.where(dh <= da, 0, 1).astype(np.int8)
    return np.where(F["ball_seen"] & (near <= cfg["obs_radius_m"]), obs, -1).astype(np.int8)


def observation_weight(F, cfg):
    """1 when one team is clearly closest and within the control radius, tapering to 0 at the
    observation radius; halved when both teams are within 1 m of each other's distance (duel)."""
    dh = np.nan_to_num(F["d_home"], nan=np.inf)
    da = np.nan_to_num(F["d_away"], nan=np.inf)
    near, far = np.minimum(dh, da), np.maximum(dh, da)
    r0, r1 = cfg["control_radius_m"], cfg["obs_radius_m"]
    w = np.clip((r1 - near) / (r1 - r0), 0, 1)
    w = np.where(far - near < cfg["duel_margin_m"], w * 0.5, w)
    return np.where(F["ball_seen"], w, 0.0)


def correlation_length(obs, graded):
    """Lag (frames) at which the autocorrelation of the +/-1 team observation drops below 1/e."""
    x = np.where(obs == 0, 1.0, np.where(obs == 1, -1.0, 0.0))[graded]
    x = x - x.mean()
    var = np.dot(x, x)
    for lag in range(1, 500):
        if np.dot(x[:-lag], x[lag:]) / var < np.exp(-1):
            return lag
    return 500


def predict(atd, F, cfg):
    n = len(F["ball_seen"])
    graded = io.graded_mask(n)
    obs = observation(F, cfg)
    info = {}
    p_dead, dinfo = deadball.p_dead(F, graded, cfg, atd["pitch"])
    info.update(dinfo)
    log_T = hmm.transitions(cfg)

    w = observation_weight(F, cfg)
    prior = None
    if cfg["use_tags"]:
        # Tag reliability measured without labels: how often the tag's side agrees with a clear
        # tracking observation (w = 1) on this match. strength = log-odds of that agreement.
        unit = tags.team_prior(io.load_smart_tagging(), n, 2.0)[:, 0]   # +1 H, -1 A, 0 none
        both = graded & (unit != 0) & (obs >= 0) & (w >= 1)
        agree = float(np.mean((unit[both] > 0) == (obs[both] == 0)))
        strength = float(np.log(agree / (1 - agree))) if agree > 0.5 else 0.0
        info["tag_agreement"] = round(agree, 3)
        info["tag_strength"] = round(strength, 3)
        st = io.load_smart_tagging()
        tag_len = float(np.median((st.end - st.start)[st.team_code >= 0]))
        info["tag_median_frames"] = tag_len
        # one tag = `strength` nats in total, spread over its frames; the emissions are later
        # divided by the temperature, so pre-multiply to keep the tag's weight independent of it.
        prior = tags.team_prior(st, n, 2 * strength / tag_len) if strength else None

    def decode(temperature):
        pr = None if prior is None else prior * temperature
        log_e = hmm.emissions(obs, p_dead, dict(cfg, temperature=temperature), w, pr)
        labels = np.full(n, "X", dtype="<U1")
        for _, a, b in half_video_blocks(GAME):
            labels[a:b + 1] = LABELS[hmm.viterbi(log_e[a:b + 1], log_T)]
        return labels

    if cfg.get("temperature"):
        t = float(cfg["temperature"])
    else:
        # Self-consistency, no labels: choose the temperature whose decoded possession rate is
        # closest to the rate measured on the sample games (possessions per 90 min of tracking).
        target = cfg["target_possessions_per_90min"]
        minutes = graded.sum() / 25 / 60
        best = None
        for t_try in cfg["temperature_grid"]:
            rate = summary(decode(t_try))["possessions"] / minutes * 90
            info.setdefault("temperature_scan", {})[t_try] = round(rate, 1)
            if best is None or abs(rate - target) < best[0]:
                best = (abs(rate - target), t_try)
        t = best[1]
    info["temperature"] = t
    info["obs_correlation_frames"] = correlation_length(obs, graded)
    return decode(t), info


def summary(labels):
    g = labels[labels != "X"]
    runs = np.flatnonzero(g[1:] != g[:-1]) + 1
    starts = np.r_[0, runs]
    vals = g[starts]
    lens = np.diff(np.r_[starts, len(g)])
    poss = lens[vals != "D"]
    return {"H": int((g == "H").sum()), "A": int((g == "A").sum()), "D": int((g == "D").sum()),
            "possessions": int(len(poss)), "mean_possession_s": round(float(poss.mean()) / 25, 2),
            "dead_balls": int((vals == "D").sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--baseline", action="store_true", help="write the smooth50 baseline instead")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--set", nargs="*", default=[], help="override config: key=value")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    atd = io.load_atd()
    F = features.build(atd)
    if args.baseline:
        labels = baseline_smooth(atd, F)
        info = {}
    else:
        labels, info = predict(atd, F, cfg)
    io.write_submission(args.out, labels)
    with open(args.out, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    print(json.dumps(dict(summary(labels), **info, sha256=sha[:16]), indent=1))
    if not args.no_validate:
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                        args.out], check=True)
    return labels


if __name__ == "__main__":
    main()
