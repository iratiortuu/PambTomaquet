import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(out):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    subprocess.run([sys.executable, "-W", "ignore", "-m", "possession.run", "--out", out],
                   check=True, env=env, cwd=ROOT, capture_output=True)
    with open(out, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_two_runs_identical_and_valid(tmp_path):
    a = _run(str(tmp_path / "a.csv"))        # validate_submission runs inside, check=True
    b = _run(str(tmp_path / "b.csv"))
    assert a == b
