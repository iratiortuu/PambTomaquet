import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from possession import io  # noqa: E402

N = 199319


def test_atd_shapes_and_ball_share():
    d = io.load_atd()
    assert d["ball"].shape == (N, 2)
    assert d["players"].shape[0] == N and d["players"].shape[1] == len(d["team"]) == 46
    share = np.isfinite(d["ball"][:, 0]).mean()
    assert abs(share - 0.569) < 0.001          # DATA.md: 113,488 / 199,319
    assert np.isfinite(d["ball"][:, 0]).sum() == 113488
    assert set(np.unique(d["team"])) == {0, 1}


def test_positions_in_metres():
    d = io.load_atd()
    x = d["players"][..., 0]
    assert np.nanmax(x) > 90 and np.nanmax(x) < 130


def test_smart_tagging():
    st = io.load_smart_tagging()
    assert len(st) == 638
    assert (st.code == "SET PIECES").sum() == 59
    ko = st[st.type == "kick off (start)"]
    assert len(ko) == 2
    # tagged kickoff windows contain the real kickoffs (starter/load_smart_tagging.py check)
    ko = ko.sort_values("start")
    for (_, r), real in zip(ko.iterrows(), (11185, 113742)):
        assert r.start <= real <= r.end


def test_graded_mask():
    m = io.graded_mask(N)
    assert m.sum() == 158857 and m[11185] and not m[11184] and m[190197] and not m[190198]
