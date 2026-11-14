"""Generate the two exploration notebooks (the reproduction does not depend on them)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nb(cells):
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s,
                       **({"outputs": [], "execution_count": None} if t == "code" else {})}
                      for t, s in cells],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                        "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = """import os, sys
ROOT = os.path.abspath('..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, matplotlib.pyplot as plt
from possession import io, features, run, deadball
atd = io.load_atd(); F = features.build(atd); st = io.load_smart_tagging()
graded = io.graded_mask(len(F['ball_seen']))"""

explore = nb([
    ("markdown", "# 01 - Exploring the tactical ATD and Smart Tagging\nInspection only: no labels, "
                 "no frame-specific constants."),
    ("code", SETUP),
    ("markdown", "## Ball-missing runs"),
    ("code", """m = ~F['ball_seen'] & graded
edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
lens = (edges[1::2] - edges[::2]) / 25
print(len(lens), 'gaps; median %.1fs, p90 %.1fs, max %.1fs' % (np.median(lens), np.percentile(lens, 90), lens.max()))
plt.hist(np.clip(lens, 0, 30), bins=60); plt.xlabel('ball-missing run (s, clipped at 30)'); plt.yscale('log'); plt.show()"""),
    ("markdown", "## Median player speed is bimodal (play vs stoppage)"),
    ("code", """x = np.log(np.clip(F['med_speed_2s'][graded], 0.05, None))
w, mu, sd = deadball.gmm2_1d(x)
plt.hist(x, bins=120, density=True)
for k in range(2):
    t = np.linspace(x.min(), x.max(), 300)
    plt.plot(t, w[k] * np.exp(-0.5 * ((t - mu[k]) / sd[k]) ** 2) / (sd[k] * np.sqrt(2 * np.pi)))
plt.xlabel('log median player speed, 2 s window'); plt.show(); print(np.exp(mu), w)"""),
    ("markdown", "## Feature traces over 10 minutes with Smart Tagging set pieces"),
    ("code", """a, b = 30000, 45000
t = np.arange(a, b) / 25 / 60
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].plot(t, F['d_home'][a:b], lw=.5, label='home'); ax[0].plot(t, F['d_away'][a:b], lw=.5, label='away')
ax[0].set_ylim(0, 15); ax[0].set_ylabel('nearest (m)'); ax[0].legend()
ax[1].plot(t, F['med_speed_2s'][a:b]); ax[1].set_ylabel('median speed')
ax[2].plot(t, F['oob'][a:b], lw=.5); ax[2].plot(t, ~F['ball_seen'][a:b] * .5, lw=.5); ax[2].set_ylabel('oob / missing')
for r in st[(st.code == 'SET PIECES') & (st.start > a) & (st.start < b)].itertuples():
    for x_ in ax: x_.axvline(r.start / 25 / 60, color='k', ls=':')
ax[2].set_xlabel('minute of video'); plt.show()"""),
])

diag = nb([
    ("markdown", "# 02 - Diagnostics of the decoded possession structure\nUses no ground truth. "
                 "Run `python -m possession.run` first."),
    ("code", SETUP + "\nimport pandas as pd\nlab = pd.read_csv(os.path.join(ROOT, 'evaluation/results/PambTomaquet.csv')).label.to_numpy()"),
    ("code", """g = lab[lab != 'X']
starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
lens = np.diff(np.r_[starts, len(g)]) / 25; vals = g[starts]
print(run.summary(lab))
plt.hist(lens[vals != 'D'], bins=np.arange(0, 120, 2), alpha=.6, label='possession')
plt.hist(lens[vals == 'D'], bins=np.arange(0, 120, 2), alpha=.6, label='dead ball')
plt.legend(); plt.xlabel('seconds'); plt.show()"""),
    ("markdown", "## Timeline of one half"),
    ("code", """a, b = 11185, 90652
c = {'H': 0, 'A': 1, 'D': 2}
y = np.array([c[x] for x in lab[a:b]])
plt.figure(figsize=(16, 2)); plt.imshow(y[None, ::5], aspect='auto', cmap='brg', interpolation='nearest')
plt.yticks([]); plt.title('1st half: H / A / D'); plt.show()"""),
    ("markdown", "## Sensitivity sweep (docs/sensitivity.txt)"),
    ("code", "print(open(os.path.join(ROOT, 'docs/sensitivity.txt')).read())"),
])

os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
for name, book in (("01_explore.ipynb", explore), ("02_diagnostics.ipynb", diag)):
    with open(os.path.join(ROOT, "notebooks", name), "w") as fh:
        json.dump(book, fh, indent=1)
print("written")
