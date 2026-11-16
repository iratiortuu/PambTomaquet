# PambTomaquet: ball possession from tracking with a calibrated HMM

More than a Hack 2027, Phase 1. Team **PambTomaquet**. Submission: `PambTomaquet.csv`.

## 1. Approach

We treat possession as a hidden state that changes rarely and is observed noisily. Each frame of
a half is in one of three states, H, A or D. A three-state hidden Markov model is decoded with
Viterbi, separately for each half. Frames outside the two halves are written as `X`. Every frame
inside a half gets an explicit H, A or D.

We chose this over frame-wise rules plus smoothing (the baseline) because the ranking metric
counts whole possessions. A majority-vote window has one knob that trades flicker against late
transitions. An HMM separates the two concerns:

- **how much to trust one frame** (emissions)
- **how often possession really changes** (transitions)

Both can be set from data that is not this match.

**Emissions (per frame)**
- *Team observation.* When the ball is seen, it is the team of the nearest tracked player, with
  P(correct) = 0.81. Each observation is weighted between 0 and 1. The weight is 1 when that
  player is inside the control radius (3.42 m), tapers to 0 at the observation radius (6.2 m),
  and is halved when both teams are within 1 m of each other's distance (a duel). It is cut to
  30% while the ball is in flight, i.e. faster than 9 m/s averaged over ±2 frames: a pass or
  clearance is not controlled by whoever it flies past. When the ball is not seen there is no team
  evidence.
- *Dead-ball evidence* p_dead. A two-component Gaussian mixture is fitted, without labels, to the
  log of the median speed of all tracked players over ±2 s on this match. The slow component's
  posterior is p_dead, clipped to [0.05, 0.95]. Seeing the ball outside the lines raises p_dead to
  0.98, and it stays there until the ball is seen moving inside the pitch faster than 4 m/s. The
  fitted modes are 0.60 m/s (13% of frames) and 1.56 m/s.
- *Smart Tagging prior.* In-possession phases (Build Up, Progression, Final Third, Set Pieces and
  others) give evidence to their tagged team. Defensive phases give it to the opponent. `N/A`
  tags, which include every throw-in, are ignored, so the tracking decides those sides. The
  strength is measured without labels: on frames with a clear tracking observation, the tag side
  agrees 73.3% of the time, so log-odds = 1.01. That total is spread over the tag's frames
  (median 565), so a phase counts as one piece of evidence and not as hundreds.

**Transitions** use the mean possession length (12.2 s), the mean dead-ball length (24.0 s) and the
share of possessions that end in a stoppage (21%). All three are measured on the sample games
(section 3).

**Temperature.** Consecutive frames are not independent: the team observation stays correlated
for about 43 frames. Emissions are therefore divided by a temperature τ. τ is the only value
chosen on this match, and the choice uses no labels. We take the τ from a fixed grid whose
decoded possession rate is closest to a target rate: 75% of the rate measured on the sample games
(314.7 per 90 minutes). The 75% comes from the proxy benchmark (section 3b). Under noisy evidence,
predicting somewhat fewer possessions than truly exist scores better on spellF1, because each
spurious split destroys a correct match. On BAR-ATH it came out at τ = 4, giving 253 possessions
(mean 20.8 s) and 65 dead-ball stretches over 106 graded minutes.

## 2. Assets used and dropped

| asset | used | how / why not |
|---|---|---|
| tactical ATD (`.xml` + `.txt`) | **yes** | ball and player positions in metres, player speeds, slot→team mapping |
| Smart Tagging | **yes** | team-phase prior with measured strength. Set pieces count only through the phase prior |
| tactical homography | no | its job is projecting to pixels, which our method does not need |
| tactical video | no | a detector to recover the 41% missing ball was out of scope for a laptop without a GPU |
| broadcast video / ATD / homography | no | not synchronised to tactical, a third has no homography, and the extra signal was not worth the alignment work |

## 3. Training and calibration data

- **No labels from the provided game were used**, in any amount, for anything. The public
  5-minute ground truth was used only to check that the pipeline is correct: frame alignment,
  label meaning and file format.
- **Calibration data:** Metrica Sports public sample games 1 and 2
  (github.com/metrica-sports/sample-data, commit `e706dd5`). Tracking for both teams plus event
  data annotated by Metrica: 190.8 minutes of tracking, 1,745 + 1,935 events. These are other
  games, as RULES.md allows. `python -m possession.calibrate` downloads them and writes
  `config/calibration.yaml` deterministically.
- A per-frame reference possession is built from those events by a written-down rule.
  On-ball events (pass, recovery, shot, ball lost, set piece, challenge won) give their team the
  ball. *Ball out* and *fault received* start a dead ball, which ends only at a restart. Loose
  balls stay with the last team. From it we measure:
  - possession and dead-ball rates and durations
  - the distance from the ball to the possessing team
  - how often the nearest team is the possessing team
- Our planning assumption that the sample ball is rarely missing was wrong. It is missing on
  40.1% of frames, close to BAR-ATH's 43%. That makes the calibration more transferable than we
  had expected.

### 3b. Proxy benchmark: how design choices were made

`python -m possession.proxy` runs the full pipeline on sample games 1 and 2 and scores it with the
official `evaluation/possession_metrics.py` against the event-derived reference. The sample
tracking is cleaner than BAR-ATH's, so `--degrade` removes ball observations in contiguous gaps
until the ball is missing on 40.7% of alive frames, as it is on BAR-ATH. The gaps are seeded,
favour frames near players (where real trackers lose the ball) and have a median of 1 s. Every
change in v2 was accepted or rejected on this benchmark alone (`docs/proxy.txt`).

| variant (degraded proxy, mean of 2 games) | spellF1 | macroF1 |
|---|---|---|
| baseline smooth50 | 0.333 | 0.679 |
| v1 (submitted first) | 0.409 | 0.679 |
| v1 + ball-in-flight down-weighting | 0.450 | 0.707 |
| **v2 = + target at 75% of the possession rate** | **0.454** | **0.702** |
| v2 on the clean (non-degraded) proxy | 0.487 | 0.709 |

**Tried and dropped** on the proxy:
- *A learned dead-ball classifier.* Logistic regression on speed, ball-seen and ball-still window
  features, trained on the sample games. Across games its frame D-F1 was 0.51–0.59, no better
  than the unsupervised detector.
- *Snapping transitions back to the onset of the new team's evidence.* It made no difference:
  Viterbi already places the switches there.

**Where the headroom is.** We ran an oracle analysis on the proxy (`scripts/oracle.py`).
Replacing our dead-ball mask with the reference lifts spellF1 to about 0.57. Replacing our team
labels with the reference while keeping our dead-ball mask lifts it to about 0.83. So the
remaining error sits mostly in *which team* has the ball during alive play, not in the
stoppages.

## 4. Possession rules implemented

- **Loose balls and contested balls** stay with the last team. In a duel the observation weight
  is halved, and the transition cost requires sustained evidence for the other team before the
  state switches.
- **Single touches, deflections and aerial duels** without follow-up control do not switch
  possession. A few frames of opposing evidence cannot pay the switching cost (see
  `tests/test_model.py::test_deflection_does_not_switch`).
- **Tracker gaps.** With the ball missing there is no team evidence, so the state carries through
  the gap without splitting the possession (see `test_gap_inside_possession_does_not_split`).
- **Restarts.** D starts when the ball is seen out, or when the whole pitch slows down. It ends
  when the ball moves again in play. The restarting team comes from the first sustained
  observation after the restart. Throw-ins have no team tag, so the tracking decides them.
- **Dead-ball ownership** is not modelled. D is always D.

## 5. Results and robustness

- On the public 5 minutes (a correctness check, *not* a ranking) v2 scores macroF1 0.705 and
  spellF1 0.459, with 17 possessions against 17 in the truth. v1 scored 0.688 and 0.400, and the
  baseline smooth50 scores 0.621 and 0.326. With 17 possessions the bootstrap noise band is
  ±0.33, so this window cannot rank anything, and we did not use it to choose between variants.
  The v2 changes were chosen on the proxy before this number was computed.
- **Sensitivity** (`docs/sensitivity.txt`, no ground truth used). Changing any calibrated
  parameter by ±20% changes under 1.3% of graded frames, with one exception: `p_nearest_correct`.
  That parameter sets the evidence per frame and moves the possession count from 162 to 434
  across ±20%. That is why τ is fixed by the possession-rate criterion and not by hand.

## 6. Limitations and what we would do next

- About 41% of graded frames have no ball, and those frames concentrate around challenges and
  boxes, which is where transitions happen. Transition timing inside long gaps is therefore
  decided by the first observation after the gap. That can be late, and spellF1 punishes lateness
  (1 s late gives about 0.65). A CPU ball detector on gap frames only, or ball-trajectory
  interpolation from player motion, is the next step.
- The dead-ball signal is player speed only. Short stoppages with players still moving, such as a
  quick throw-in, are hard to see. Set-piece tags could anchor restarts more precisely.
- The observation is "nearest player", not "controlling player". A receiver-side model that looks
  at ball deceleration near a player would fix transitions on long passes.
- The calibration games are cleaner and older than a LaLiga 2024-25 match, so the rates may differ.
- The proxy reference comes from events, while the challenge truth comes from corrected tracking.
  The two may draw possession and dead-ball boundaries differently, so a proxy gain is evidence
  of a BAR-ATH gain, not proof of one.
- Development history: we looked at public-window scores while debugging. One change was made
  after seeing them: tag evidence had been counted once per frame, so a 10-second phase counted
  as 250 votes. We fixed that double-counting. No parameter was set from those scores.

## 7. Reproduction

```bash
git clone https://github.com/iratiortuu/PambTomaquet && cd PambTomaquet
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-pambtomaquet.txt
# place the match bundle in ./data (see data/README.md) or export MTAH_DATA_DIR=...
PYTHONPATH=src .venv/bin/python -m possession.calibrate   # optional: rebuilds config/calibration.yaml
PYTHONPATH=src .venv/bin/python -m possession.run          # -> evaluation/results/PambTomaquet.csv
.venv/bin/python -m pytest -q tests                        # includes a two-run SHA256 check
```

The run takes about a minute on a laptop CPU once the tracking cache is built. Expected output
SHA-256 begins with `4900d84e93d13d99`.

`python -m possession.proxy --degrade` reproduces the proxy benchmark.
