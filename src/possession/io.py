"""Load the tactical ATD and Smart Tagging into numpy/pandas, indexed by VIDEO frame.

Video frame = ATD frame_id - 1. Positions are converted from normalised 0-1 to metres using the
pitch size carried by games.py (105.3 x 68.0). Parsed tracking is cached to cache/*.npz keyed on
the raw file's size and mtime, so a second run takes a second rather than a minute.
"""

import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import CACHE_DIR
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402

HOME, AWAY = 0, 1


def team_of_slot(meta_path):
    """0 = home, 1 = away for every track slot. First <Team> in the EPTS metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}".format(meta_path))
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _parse_raw(raw_path, n_frames, n_slots, pitch):
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n_frames:
                continue
            vals = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            vals = vals.reshape(-1, 3)[:n_slots]
            players[f, :len(vals)] = vals[:, :2]
            speed[f, :len(vals)] = vals[:, 2]
            b = parts[2].split(",")
            if b[0][0] != "N":
                ball[f] = (float(b[0]), float(b[1]))
                ball_speed[f] = float(b[2]) if len(b) > 2 and b[2][0] != "N" else np.nan
    scale = np.array([length, width], dtype=np.float32)
    return {"ball": ball * scale, "ball_speed": ball_speed,
            "players": players * scale, "speed": speed}


def load_atd(game=GAME, data_dir=None, use_cache=True):
    """Dict of arrays: ball[n,2] m, ball_speed[n], players[n,S,2] m, speed[n,S], team[S]."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_slot(meta_path)
    n = source_frames("tactical", game)
    st = os.stat(raw_path)
    key = hashlib.sha1("{}|{}|{}".format(os.path.basename(raw_path), st.st_size,
                                          int(st.st_mtime)).encode()).hexdigest()[:12]
    cache = os.path.join(CACHE_DIR, "atd_tactical_{}.npz".format(key))
    if use_cache and os.path.exists(cache):
        with np.load(cache) as z:
            data = {k: z[k] for k in z.files}
    else:
        data = _parse_raw(raw_path, n, len(team), GAMES[game]["pitch"])
        data["team"] = team
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **data)
    data["pitch"] = np.array(GAMES[game]["pitch"], dtype=np.float32)
    return data


def load_smart_tagging(game=GAME, data_dir=None):
    """One row per instance: code, start, end (video frames, end exclusive), Team, Type, Half."""
    path = require(["smart_tagging"], game, data_dir)[0]
    fps = GAMES[game]["fps"]
    rows = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        s, e = float(inst.findtext("start")), float(inst.findtext("end"))
        rows.append({"code": (inst.findtext("code") or "").strip().upper(),
                     "start": int(round(s * fps)), "end": int(round(e * fps)),
                     "team": (tags.get("Team") or "").strip().lower(),
                     "type": (tags.get("Type") or "").strip().lower(),
                     "half": (tags.get("Half") or "").strip().lower()})
    df = pd.DataFrame(rows).sort_values(["start", "end", "code"], kind="stable")
    tmap = {"barcelona": HOME, "atletic club": AWAY}
    df["team_code"] = df["team"].map(tmap).fillna(-1).astype(int)
    return df.reset_index(drop=True)


def graded_mask(n, game=GAME):
    m = np.zeros(n, dtype=bool)
    for _, a, b in half_video_blocks(game):
        m[a:b + 1] = True
    return m


def write_submission(path, labels):
    """frame,label CSV with '\\n' endings -- byte-stable across runs and platforms."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path
