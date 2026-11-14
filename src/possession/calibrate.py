"""Derive possession hyperparameters from Metrica's PUBLIC sample games 1 and 2.

These are other games (anonymised, Metrica Sports, https://github.com/metrica-sports/sample-data,
human-annotated event data by Metrica). Nothing here touches BAR-ATH. The output is
config/calibration.yaml; every number in it is reproducible by

    python -m possession.calibrate

Possession reference built from the events (a mechanical rule, written down here so it is
auditable):
  * on-ball events (PASS, RECOVERY, SHOT, BALL LOST, SET PIECE, CHALLENGE *-WON) give their team
    the ball from their start frame;
  * BALL OUT and FAULT RECEIVED make the ball dead from their frame until the next SET PIECE
    (or the next PASS by a goalkeeper / goal kick, which Metrica tags as PASS GOAL KICK);
  * otherwise possession persists (loose balls stay with the last team).
The sample tracking is at 25 fps like ours; frames are aligned directly.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import yaml

from . import EXTERNAL_DIR, ROOT

SAMPLE_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"   # pinned for reproducibility
BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/data".format(SAMPLE_COMMIT)
GAMES = (1, 2)
PITCH = (105.0, 68.0)   # Metrica sample games are normalised on a 105 x 68 pitch
FPS = 25
OUT = os.path.join(ROOT, "config", "calibration.yaml")

ON_BALL = {"PASS", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE"}
DEAD_START = {"BALL OUT", "FAULT RECEIVED"}


def _fetch(game, name):
    folder = os.path.join(EXTERNAL_DIR, "sample-data")
    os.makedirs(folder, exist_ok=True)
    fname = "Sample_Game_{}_{}.csv".format(game, name)
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        url = "{}/Sample_Game_{}/{}".format(BASE, game, fname)
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    return path


def _tracking(game, side):
    df = pd.read_csv(_fetch(game, "RawTrackingData_{}_Team".format(side)), skiprows=2)
    cols = list(df.columns)
    xy = []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            xy.append(df.iloc[:, [i, i + 1]].to_numpy(np.float64))
    players = np.stack(xy, axis=1) * PITCH            # [n, p, 2] metres
    bi = cols.index("Ball")
    ball = df.iloc[:, [bi, bi + 1]].to_numpy(np.float64) * PITCH
    return df["Frame"].to_numpy(int), df["Period"].to_numpy(int), players, ball


def reference_labels(events, n):
    """Per-frame 0 home / 1 away / 2 dead / -1 before kickoff, from the event rules above."""
    lab = np.full(n + 2, -1, dtype=np.int8)
    ev = events.sort_values("Start Frame", kind="stable")
    state, last = -1, 0
    marks = []
    for _, e in ev.iterrows():
        t, typ = str(e["Type"]), str(e["Subtype"])
        team = 0 if e["Team"] == "Home" else 1
        f = int(e["Start Frame"])
        if typ == "nan":
            typ = ""
        if t in DEAD_START:
            marks.append((f, 2))
        elif t in ON_BALL or (t == "CHALLENGE" and typ.endswith("WON") and "FAULT" not in typ):
            if state == 2 and t not in ("SET PIECE",) and "GOAL KICK" not in typ \
                    and t != "PASS":
                continue          # a dead ball only ends at a restart
            marks.append((f, team))
            state = team
        else:
            continue
        if marks[-1][1] == 2:
            state = 2
    state = -1
    marks.sort(key=lambda m: m[0])
    for f, s in marks:
        f = min(max(f, 0), n + 1)
        lab[last:f] = state
        last, state = f, s
    lab[last:] = state
    return lab[1:n + 1]                               # frame k -> index k-1


def _runs(x):
    if len(x) == 0:
        return np.array([], int), np.array([], int), np.array([], int)
    edges = np.flatnonzero(np.diff(x)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(x)]
    return starts, ends, x[starts]


def analyse_game(game):
    events = pd.read_csv(_fetch(game, "RawEventsData"))
    frames, period, home, ball = _tracking(game, "Home")
    _, _, away, _ = _tracking(game, "Away")
    n = len(frames)
    lab = reference_labels(events, n)

    s, e, v = _runs(lab)
    dur = (e - s) / FPS
    poss = dur[(v == 0) | (v == 1)]
    dead = dur[v == 2]
    # possessions split by dead balls or team changes only (H D H = 2): maximal runs
    alive = (lab == 0) | (lab == 1)

    d_home = np.nanmin(np.linalg.norm(home - ball[:, None], axis=2), axis=1)
    d_away = np.nanmin(np.linalg.norm(away - ball[:, None], axis=2), axis=1)
    d_own = np.where(lab == 0, d_home, np.where(lab == 1, d_away, np.nan))
    d_opp = np.where(lab == 0, d_away, np.where(lab == 1, d_home, np.nan))
    nearest_team = np.where(d_home <= d_away, 0, 1)
    ok = alive & np.isfinite(d_home) & np.isfinite(d_away)
    agree = float(np.mean(nearest_team[ok] == lab[ok]))

    # Distance of the gaining team at the frame possession changes hands.
    change = np.flatnonzero(alive[1:] & alive[:-1] & (lab[1:] != lab[:-1])) + 1
    d_gain = np.where(lab[change] == 0, d_home[change], d_away[change])

    # "Opponent episodes": runs where the non-possessing team's nearest player is within r of
    # the ball. Those that end in a change vs those that do not -> dwell threshold k.
    ball_speed = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
    return {
        "minutes": n / FPS / 60.0, "poss": poss, "dead": dead, "d_own": d_own[ok],
        "d_opp": d_opp[ok], "d_gain": d_gain[np.isfinite(d_gain)], "agree": agree,
        "lab": lab, "d_home": d_home, "d_away": d_away, "ball_speed": ball_speed,
        "ball_missing": float(np.mean(~np.isfinite(ball[:, 0]))),
        "events": events,
    }


def dwell_separation(games, radius):
    """For each opponent-near episode (opponent within `radius` m for >=1 frame while ball alive),
    record its length and whether possession changed to that team during/just after it."""
    rows = []
    for g in games:
        lab, dh, da = g["lab"], g["d_home"], g["d_away"]
        alive = (lab == 0) | (lab == 1)
        for team, d in ((0, dh), (1, da)):
            near = alive & (lab != team) & (d < radius)
            near_any = near | (alive & (lab == team) & (d < radius))
            s, e, v = _runs(near_any.astype(np.int8))
            for a, b, val in zip(s, e, v):
                if not val or not near[a]:
                    continue
                won = bool(np.any(lab[a:min(b + 12, len(lab))] == team))
                rows.append((b - a, won))
    return np.array(rows, dtype=float)


def pct(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"p{}".format(q): round(float(np.percentile(x, q)), 3) for q in qs}


def main():
    games = [analyse_game(g) for g in GAMES]
    minutes = sum(g["minutes"] for g in games)
    poss = np.concatenate([g["poss"] for g in games])
    dead = np.concatenate([g["dead"] for g in games])
    d_own = np.concatenate([g["d_own"] for g in games])
    d_opp = np.concatenate([g["d_opp"] for g in games])
    d_gain = np.concatenate([g["d_gain"] for g in games])

    control_radius = round(float(np.percentile(d_own, 75)), 2)
    ep = dwell_separation(games, control_radius)
    # smallest dwell k (frames) at which an opponent episode is more likely than not a change
    k_best = None
    for k in range(1, 76):
        sel = ep[ep[:, 0] >= k]
        if len(sel) >= 20 and sel[:, 1].mean() >= 0.5:
            k_best = k
            break

    cal = {
        "source": "Metrica Sports sample-data games {} (github.com/metrica-sports/sample-data, "
                  "branch {}); human event annotation by Metrica".format(list(GAMES), SAMPLE_COMMIT),
        "minutes_of_tracking": round(minutes, 1),
        "possessions_per_90min": round(len(poss) / minutes * 90, 1),
        "possession_duration_s": dict(pct(poss), mean=round(float(poss.mean()), 2)),
        "dead_ball_per_90min": round(len(dead) / minutes * 90, 1),
        "dead_ball_duration_s": dict(pct(dead), mean=round(float(dead.mean()), 2)),
        "dist_ball_to_possessing_team_m": pct(d_own),
        "dist_ball_to_other_team_m": pct(d_opp),
        "dist_gaining_team_at_change_m": pct(d_gain),
        "nearest_team_equals_possession_share": round(float(np.mean([g["agree"] for g in games])), 3),
        "ball_missing_share_sample": round(float(np.mean([g["ball_missing"] for g in games])), 3),
        "control_radius_m": control_radius,
        "control_dwell_frames": k_best,
        "opponent_episodes": int(len(ep)),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        yaml.safe_dump(cal, fh, sort_keys=False)
    print(yaml.safe_dump(cal, sort_keys=False))
    return cal


if __name__ == "__main__":
    main()
