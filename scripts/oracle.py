"""Where is the headroom? On the degraded sample-game proxy (other games), replace one half of our
prediction with the reference and re-score: (a) reference dead-ball mask + our teams,
(b) our dead-ball mask + reference teams. No BAR-ATH labels involved."""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from possession import features, proxy, run  # noqa: E402

cfg = yaml.safe_load(open(run.CONFIG))
cfg["temperature"] = 3
for game in (1, 2):
    atd, ref, blocks = proxy.load_game(game)
    atd = proxy.degrade(atd, ref, np.random.default_rng(game))
    F = features.build(atd)
    lab, _ = run.predict(atd, F, dict(cfg), blocks=blocks, smart_tagging=None)
    R = np.array(["H", "A", "D", "X"])[np.where(ref < 0, 3, ref)]
    ours_alive_team = lab.copy()
    a = np.where(R == "D", "D", np.where(lab == "D", R, lab))           # true D, our team
    b = np.where(lab == "D", "D", np.where(R == "D", lab, R))           # our D, true team
    b = np.where((lab != "D") & (R == "D"), ours_alive_team, b)
    for name, x in (("ours", lab), ("trueD+ourTeam", a), ("ourD+trueTeam", b)):
        x = np.where(lab == "X", "X", x)
        r = proxy.score(x, ref, blocks)
        print(game, "{:<16} spellF1 {:.4f}  poss {}/{}".format(name, r["spell_f1_mean"],
                                                               r["n_pred_spells"], r["n_gt_spells"]))
