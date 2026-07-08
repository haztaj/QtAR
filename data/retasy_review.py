#!/usr/bin/env python3
"""
Stage 2 of the RetaSy cleanup — a static by-ear review page over the auto-flags.

Same pattern as the waqf audition page (data/segment_waqf.py): a single self-contained
HTML file, no server. Reads data/raw/retasy_audio/flags.csv (Stage 1), groups clips by
bucket with the borderline band FIRST, and for each clip shows the audio, the labeled
ayah's Arabic text, the decoded phonemes, and the match costs — plus keep / discard /
relabel buttons. A "Download verdicts" button exports review_verdicts.json; move it to
data/retasy_verdicts.json (committed, reproducible cleanup) and make_phase2_splits.py
picks it up (Stage 3).

Time-savers so you only judge the middle band:
  - The extremes are PRE-VERDICTED: dead-silent (silent) + clear garbage default to
    discard; high-confidence ok defaults to keep. A small random sample of each is
    surfaced (--spot N) so you can spot-check the auto-verdicts.
  - possible_mislabel clips pre-fill the suggested ayah in the relabel box.

Audio is referenced by absolute file:// path (RetaSy wavs are not copied — clips stay in
data/raw, never committed). Open the HTML in a browser on this machine.

  python data/retasy_review.py                     # all flagged, extremes auto-verdicted
  python data/retasy_review.py --spot 30           # surface 30 of each auto-verdicted class
  python data/retasy_review.py --only borderline,possible_mislabel,garbage,noise_only
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FLAGS = REPO / "data" / "raw" / "retasy_audio" / "flags.csv"
AYAH_TEXT = REPO / "data" / "manifests" / "ayah_text.json"
OUT_HTML = REPO / "data" / "raw" / "retasy_audio" / "review.html"

# review order: the human-judgement band first; auto-verdicted extremes last (spot-check).
# `too_short` is shown IN FULL (not spot-sampled): it's small (~124) and it's where
# genuinely-good-but-brief learner clips leak into auto-discard (validated on the labeled
# subset — most good-clip false-discards are too_short). Default there is discard, so the
# human only has to flip the few worth keeping.
BUCKET_ORDER = ["possible_mislabel", "borderline", "too_short", "garbage", "noise_only",
                "silent", "ok"]
AUTO_DISCARD = {"silent", "garbage", "noise_only", "too_short"}
AUTO_KEEP = {"ok"}
FULL_REVIEW = {"possible_mislabel", "borderline", "too_short"}   # shown in full, not sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", type=Path, default=FLAGS)
    ap.add_argument("--out", type=Path, default=OUT_HTML)
    ap.add_argument("--spot", type=int, default=25,
                    help="clips per auto-verdicted bucket to surface for spot-checking")
    ap.add_argument("--only", default="",
                    help="comma-separated buckets to include (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.flags, encoding="utf-8").fillna("")
    text = json.loads(AYAH_TEXT.read_text(encoding="utf-8")) if AYAH_TEXT.exists() else {}
    only = set(args.only.split(",")) if args.only else None

    # decide which clips to actually render: full manual buckets + a spot sample of the rest
    manual = FULL_REVIEW
    render = []
    for b in BUCKET_ORDER:
        if only and b not in only:
            continue
        sub = df[df["bucket"] == b]
        if b not in manual and only is None:
            sub = sub.sample(min(len(sub), args.spot), random_state=args.seed) if len(sub) else sub
        render.append((b, sub))

    n_render = sum(len(s) for _, s in render)
    n_auto_keep = int(df["bucket"].isin(AUTO_KEEP).sum())
    n_auto_discard = int(df["bucket"].isin(AUTO_DISCARD).sum())

    sections = []
    for b, sub in render:
        if not len(sub):
            continue
        default = ("discard" if b in AUTO_DISCARD else "keep" if b in AUTO_KEEP else "")
        rows = []
        for r in sub.itertuples():
            arab = text.get(r.key, "")
            sugg_arab = text.get(r.suggest, "") if r.suggest else ""
            src = Path(r.path).as_uri()
            checked = {"keep": "", "discard": "", "relabel": ""}
            if default:
                checked[default] = "checked"
            rows.append(
                f"<tr data-id='{html.escape(str(r.recording_id))}' data-suggest='{html.escape(str(r.suggest))}'>"
                f"<td class='k'>{html.escape(r.key)}</td>"
                f"<td><audio controls preload='none' src='{html.escape(src)}'></audio><br>"
                f"<small>{r.duration:.1f}s · rms {r.rms} · speech {r.speech_frac} ({r.speech_sec}s)</small></td>"
                f"<td dir='rtl' style='font-size:20px'>{html.escape(arab)}"
                f"<div class='ph'>{html.escape(str(r.n_phon))} ph · label-cost {r.label_cost}</div></td>"
                f"<td>{('best <b>' + html.escape(str(r.best_key)) + '</b> @ ' + str(r.best_cost)) if r.best_key else ''}"
                + (f"<div dir='rtl' style='font-size:18px;color:#0a0'>{html.escape(sugg_arab)}</div>" if sugg_arab else "")
                + (f"<div class='fl'>human: {html.escape(str(r.final_label))}</div>" if r.final_label else "")
                + "</td>"
                f"<td class='v'>"
                f"<label><input type='radio' name='v_{r.recording_id}' value='keep' {checked['keep']}>keep</label>"
                f"<label><input type='radio' name='v_{r.recording_id}' value='discard' {checked['discard']}>discard</label>"
                f"<label><input type='radio' name='v_{r.recording_id}' value='relabel' {checked['relabel']}>relabel"
                f"<input class='rl' name='rl_{r.recording_id}' value='{html.escape(str(r.suggest))}' size='7' placeholder='S:A'></label>"
                f"</td></tr>")
        sections.append(
            f"<h2>{b} <small>({len(sub)}"
            + (f" of {int((df['bucket'] == b).sum())} sampled — spot-check" if b not in manual and only is None else "")
            + (f", default <b>{default}</b>" if default else "") + ")</small></h2>"
            f"<table><tr><th>ayah</th><th>audio</th><th>labeled text</th><th>best match</th><th>verdict</th></tr>"
            + "".join(rows) + "</table>")

    script = """
<script>
function collect() {
  const out = {keep: [], discard: [], relabel: {}};
  document.querySelectorAll('tr[data-id]').forEach(tr => {
    const id = tr.dataset.id;
    const v = tr.querySelector('input[type=radio]:checked');
    if (!v) return;
    if (v.value === 'keep') out.keep.push(id);
    else if (v.value === 'discard') out.discard.push(id);
    else {
      const rl = tr.querySelector('input.rl').value.trim();
      if (/^\\d+:\\d+$/.test(rl)) out.relabel[id] = rl; else out.discard.push(id);
    }
  });
  return out;
}
function counts() {
  const o = collect();
  document.getElementById('counts').textContent =
    `keep ${o.keep.length} · discard ${o.discard.length} · relabel ${Object.keys(o.relabel).length}`;
}
function download() {
  const blob = new Blob([JSON.stringify(collect(), null, 1)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'review_verdicts.json'; a.click();
}
document.addEventListener('change', counts);
window.addEventListener('load', counts);
</script>"""

    page = (
        "<!doctype html><meta charset='utf-8'><title>RetaSy review</title>"
        "<style>body{font-family:sans-serif;max-width:1200px;margin:16px auto}"
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}"
        "td,th{border:1px solid #ccc;padding:6px;vertical-align:top;font-size:14px}"
        ".k{font-weight:bold;white-space:nowrap}.ph,.fl{color:#888;font-size:12px}"
        ".v label{display:block}.rl{margin-left:4px}"
        "#bar{position:sticky;top:0;background:#fff;border-bottom:2px solid #333;padding:10px 0;z-index:9}"
        "audio{width:220px}</style>"
        "<div id='bar'><b>RetaSy cleanup review</b> — "
        f"{n_render} clips shown; auto: {n_auto_keep} keep / {n_auto_discard} discard "
        "(spot-check samples below). <span id='counts'></span> "
        "<button onclick='download()'>Download verdicts</button>"
        "<p style='margin:4px 0;color:#555'>Judge <b>possible_mislabel</b> (is the green suggestion right? "
        "relabel) and <b>borderline</b> (keep if it's the labeled ayah, even mispronounced). "
        "Everything else is pre-verdicted — flip any you disagree with.</p></div>"
        + "".join(sections) + script)
    args.out.write_text(page, encoding="utf-8")

    print(f"wrote {args.out.relative_to(REPO)}  ({n_render} clips rendered)")
    print(f"  full review band: {', '.join(sorted(FULL_REVIEW))} = "
          f"{int(df['bucket'].isin(FULL_REVIEW).sum())} clips")
    print(f"  auto-verdicted: {n_auto_keep} keep / {n_auto_discard} discard "
          f"({args.spot} of each spot-checked)")
    print(f"\nopen: {args.out.as_uri()}")


if __name__ == "__main__":
    main()
