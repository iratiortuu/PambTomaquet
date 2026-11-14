"""3-state (H, A, D) hidden Markov model decoded with Viterbi, per half.

Emissions (log space, all frames of a half):
  * team observation: the team of the nearest tracked player when the ball is seen and that
    player is within `obs_radius_m`. P(obs = state team) = `p_nearest_correct`, calibrated on the
    sample games (share of alive frames where nearest team = possessing team). Under D the team
    observation is uninformative.
  * dead-ball evidence p_dead in (0, 1) from deadball.py: emission p_dead for D, 1 - p_dead for
    H and A.
Both emissions are divided by `temperature`, the correlation length of consecutive frames:
neighbouring frames are not independent observations, and treating them as such makes the
decoder switch on every flicker.

Transitions come from calibrated rates: the expected length of a possession run and of a dead
ball, expressed per frame. Ties in Viterbi are broken by state order (H, A, D), deterministically.
"""

import numpy as np

H, A, D = 0, 1, 2


def emissions(obs_team, p_dead, cfg, obs_weight=None, tag_prior=None):
    """obs_weight in [0, 1] scales each team observation (1 = clear control, 0 = no evidence).
    tag_prior[n, 2] is an additive log-evidence for H and A from Smart Tagging phases."""
    p = cfg["p_nearest_correct"]
    tau = cfg["temperature"]
    n = len(obs_team)
    le = np.zeros((n, 3))
    seen = obs_team >= 0
    w = np.ones(n) if obs_weight is None else obs_weight
    for s in (H, A):
        le[seen, s] += w[seen] * np.where(obs_team[seen] == s, np.log(p), np.log(1 - p))
    le[seen, D] += w[seen] * np.log(0.5)
    if tag_prior is not None:
        le[:, H] += tag_prior[:, 0]
        le[:, A] += tag_prior[:, 1]
    pd_ = np.clip(p_dead, 1e-4, 1 - 1e-4)
    le[:, D] += np.log(pd_)
    le[:, H] += np.log(1 - pd_)
    le[:, A] += np.log(1 - pd_)
    return le / tau


def transitions(cfg, fps=25):
    poss_f = cfg["mean_possession_s"] * fps
    dead_f = cfg["mean_dead_s"] * fps
    alive_to_dead = cfg["dead_share_of_possession_ends"] / poss_f
    alive_switch = (1 - cfg["dead_share_of_possession_ends"]) / poss_f
    dead_exit = 1.0 / dead_f
    T = np.array([
        [1 - alive_switch - alive_to_dead, alive_switch, alive_to_dead],
        [alive_switch, 1 - alive_switch - alive_to_dead, alive_to_dead],
        [dead_exit / 2, dead_exit / 2, 1 - dead_exit],
    ])
    return np.log(T)


def viterbi(log_e, log_T, log_start=None):
    n, k = log_e.shape
    if log_start is None:
        log_start = np.log(np.full(k, 1.0 / k))
    back = np.zeros((n, k), dtype=np.int8)
    score = log_start + log_e[0]
    for t in range(1, n):
        cand = score[:, None] + log_T            # [from, to]
        back[t] = np.argmax(cand, axis=0)        # first max wins -> deterministic
        score = cand[back[t], np.arange(k)] + log_e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
