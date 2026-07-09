"""Rebuild the full-clip decode caches WITH posterior alternatives (Phase 0).
Overwrites full_streams_test.pkl (+ unseg via continuous_eval) so the matcher can
read per-phoneme top-k posteriors. Run: python research/rebuild_cache.py [--unseg]"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in ("research", "training", "matcher"):
    sys.path.insert(0, str(REPO / p))

if "--unseg" in sys.argv:
    from continuous_eval import build_unseg_cache
    build_unseg_cache()
else:
    from chain_decoder import build_cache
    build_cache()
