import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from possession import deadball, features, hmm  # noqa: E402

CFG = {"p_nearest_correct": 0.8, "temperature": 3.0, "mean_possession_s": 12.0,
       "mean_dead_s": 24.0, "dead_share_of_possession_ends": 0.2}


def test_viterbi_recovers_noisy_sequence():
    rng = np.random.default_rng(0)
    truth = np.r_[np.zeros(300), np.ones(400), np.full(300, 2), np.zeros(300)].astype(int)
    obs = np.where(truth == 2, -1, truth)
    flip = rng.random(len(obs)) < 0.2
    obs = np.where(flip & (obs >= 0), 1 - obs, obs).astype(np.int8)
    p_dead = np.where(truth == 2, 0.9, 0.1)
    path = hmm.viterbi(hmm.emissions(obs, p_dead, CFG), hmm.transitions(CFG))
    assert (path == truth).mean() > 0.97
    runs = 1 + np.count_nonzero(np.diff(path))
    assert runs == 4                                   # no flicker-induced extra possessions


def test_deflection_does_not_switch():
    obs = np.zeros(600, dtype=np.int8)
    obs[300:306] = 1                                   # a 6-frame touch by the other team
    path = hmm.viterbi(hmm.emissions(obs, np.full(600, 0.05), CFG), hmm.transitions(CFG))
    assert (path == 0).all()


def test_gap_inside_possession_does_not_split():
    obs = np.zeros(900, dtype=np.int8)
    obs[300:600] = -1                                  # ball lost by the tracker for 12 s
    path = hmm.viterbi(hmm.emissions(obs, np.full(900, 0.05), CFG), hmm.transitions(CFG))
    assert (path == 0).all()


def test_viterbi_deterministic():
    rng = np.random.default_rng(1)
    obs = rng.integers(-1, 2, 2000).astype(np.int8)
    le = hmm.emissions(obs, rng.random(2000), CFG)
    assert (hmm.viterbi(le, hmm.transitions(CFG)) == hmm.viterbi(le, hmm.transitions(CFG))).all()


def test_out_of_bounds_and_hold():
    ball = np.array([[50, 30], [106.5, 30], [np.nan, np.nan], [104, 30], [100, 30]], np.float32)
    oob = features.out_of_bounds(ball, (105.3, 68.0))
    assert oob.tolist() == [False, True, False, False, False]
    speed = np.array([np.nan, 1, np.nan, 0.5, 10.0])
    hold = deadball.oob_hold(oob, np.isfinite(ball[:, 0]), speed, ball, None, 4.0)
    assert hold.tolist() == [False, True, True, True, False]


def test_gmm_separates_two_modes():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(-1, 0.2, 300), rng.normal(1, 0.2, 700)]
    w, mu, _ = deadball.gmm2_1d(x)
    assert abs(mu[0] + 1) < 0.1 and abs(mu[1] - 1) < 0.1 and abs(w[0] - 0.3) < 0.05
