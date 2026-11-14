"""PambTomaquet -- ball possession for More than a Hack 2027, Phase 1."""

import os
import sys

# The challenge repo's games.py lives at the repository root; make it importable.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(ROOT, "cache")
EXTERNAL_DIR = os.path.join(ROOT, "external")
