"""Unsupervised dead-ball evidence.

When the ball is dead the whole pitch slows down: players walk back, wait for a throw-in, line up
a wall. The median speed of all tracked players, averaged over +/- 2 s, is therefore bimodal over
a match. We fit a two-component Gaussian mixture to log(median speed) over the graded frames of
THIS match -- no labels are involved, only the unlabelled tracking -- and take the posterior of the
slow component as p_dead.

On top of that:
  * the ball seen outside the pitch lines (with a small margin) is near-certain dead ball, and
    stays dead until the ball is seen moving inside the pitch again;
  * a Smart Tagging set piece window means a restart happens inside it; we use it as mild
    evidence for dead ball at the start of the window only (the tags are automatic and noisy).
"""

import numpy as np


def gmm2_1d(x, iters=200, seed_quantiles=(0.2, 0.8)):
    """Deterministic 2-component 1-D Gaussian mixture by EM. Returns (w, mu, sd) sorted by mu."""
    x = x[np.isfinite(x)]
    mu = np.quantile(x, seed_quantiles)
    sd = np.array([x.std(), x.std()]) / 2
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
        ll -= ll.max(axis=1, keepdims=True)
        r = np.exp(ll)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0)
        w = nk / len(x)
        mu = (r * x[:, None]).sum(axis=0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk) + 1e-6
    o = np.argsort(mu)
    return w[o], mu[o], sd[o]


def posterior_slow(x, params):
    w, mu, sd = params
    ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
    ll -= np.nanmax(ll, axis=1, keepdims=True)
    r = np.exp(ll)
    return r[:, 0] / r.sum(axis=1)


def oob_hold(oob, ball_seen, ball_speed, ball, pitch, restart_speed):
    """True from a frame the ball is seen out until it is next seen moving inside the pitch."""
    n = len(oob)
    out = np.zeros(n, dtype=bool)
    inside = ball_seen & ~oob
    moving_in = inside & (np.nan_to_num(ball_speed) > restart_speed)
    dead = False
    for i in range(n):
        if oob[i]:
            dead = True
        elif dead and moving_in[i]:
            dead = False
        out[i] = dead
    return out


def p_dead(F, graded, cfg, pitch, tags=None):
    x = np.log(np.clip(F["med_speed_2s"], 0.05, None))
    params = gmm2_1d(x[graded])
    p = posterior_slow(x, params)
    p = np.where(np.isfinite(p), p, 0.5)
    p = cfg["stoppage_floor"] + (cfg["stoppage_ceiling"] - cfg["stoppage_floor"]) * p
    hold = oob_hold(F["oob"], F["ball_seen"], F["ball_speed"], None, pitch, cfg["restart_ball_speed"])
    p = np.where(hold, np.maximum(p, cfg["p_dead_oob"]), p)
    return p, {"gmm_weights": params[0].tolist(),
               "gmm_speed_ms": np.exp(params[1]).tolist(), "oob_hold_frames": int(hold.sum())}
