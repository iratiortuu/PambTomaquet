"""Per-frame features from the tactical ATD. Vectorised and NaN-safe."""

import warnings

import numpy as np

from .io import HOME, AWAY


def nearest_by_team(ball, players, team):
    """Distance (m) from the ball to the nearest tracked player of each team; NaN if unknown."""
    d = np.linalg.norm(players - ball[:, None, :], axis=2)          # [n, S]
    out = np.full((len(ball), 2), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in (HOME, AWAY):
            out[:, t] = np.nanmin(d[:, team == t], axis=1)
    return out


def run_lengths_since(mask):
    """For each frame, how many consecutive frames (including this one) mask has been True."""
    out = np.zeros(len(mask), dtype=np.int32)
    c = 0
    for i, m in enumerate(mask):
        c = c + 1 if m else 0
        out[i] = c
    return out


def ball_velocity(ball, fps=25):
    v = np.full_like(ball, np.nan)
    v[1:] = (ball[1:] - ball[:-1]) * fps
    return np.linalg.norm(v, axis=1)


def out_of_bounds(ball, pitch, margin=0.5):
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > pitch[0] + margin) | (y < -margin) | (y > pitch[1] + margin)


def team_activity(players, speed, team):
    """Median player speed over all tracked players, and per-team count tracked."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(speed, axis=1)
    n_tracked = np.isfinite(players[..., 0]).sum(axis=1)
    return med, n_tracked


def rolling_mean(x, half):
    """Centred mean over +/- half frames, ignoring NaN."""
    v = np.where(np.isfinite(x), x, 0.0)
    c = np.isfinite(x).astype(float)
    cv = np.r_[0, np.cumsum(v)]
    cc = np.r_[0, np.cumsum(c)]
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    n = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cv[hi] - cv[lo]) / n, np.nan)


def build(atd):
    ball, players, team = atd["ball"], atd["players"], atd["team"]
    near = nearest_by_team(ball, players, team)
    med_speed, n_tracked = team_activity(players, atd["speed"], team)
    return {
        "ball_seen": np.isfinite(ball[:, 0]),
        "d_home": near[:, HOME], "d_away": near[:, AWAY],
        "ball_speed": ball_velocity(ball),
        "oob": out_of_bounds(ball, atd["pitch"]),
        "med_speed": med_speed,
        "med_speed_2s": rolling_mean(med_speed, 25),
        "n_tracked": n_tracked,
    }
