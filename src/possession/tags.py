"""Smart Tagging -> weak per-frame team evidence.

The Team tag names the team performing the phase. For defensive phases that is the team WITHOUT
the ball, so the evidence goes to the opponent. Set pieces name the team taking the restart.
Throw-ins and other N/A-tagged instances carry no team and are ignored here; the tracking decides
their side. The tags are automatic, so the evidence is a small log-odds per frame, not a label.
"""

import numpy as np

DEFENSIVE = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")
IN_POSSESSION = ("BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
                 "ATTACKING TRANSITION", "LONG BALL", "BALL IN THE BOX", "SET PIECES")


def team_prior(st, n, strength):
    """[n, 2] additive log-evidence for (H, A)."""
    votes = np.zeros((n, 2))
    for r in st.itertuples():
        if r.team_code < 0:
            continue
        if r.code in DEFENSIVE:
            side = 1 - r.team_code
        elif r.code in IN_POSSESSION:
            side = r.team_code
        else:
            continue
        a, b = max(r.start, 0), min(r.end, n)
        votes[a:b, side] += 1
    # Only the net vote matters, and it is capped at one tag's worth: overlapping phases from the
    # same automatic tagger are not independent evidence.
    net = np.clip(votes[:, 0] - votes[:, 1], -1, 1)
    prior = np.zeros((n, 2))
    prior[:, 0] = strength * net / 2
    prior[:, 1] = -strength * net / 2
    return prior
