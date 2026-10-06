"""Precompute the showcase documents so the demo shows them instantly (no model loading).

Runs the live pipeline once on a few held-out test summaries of increasing difficulty and
writes text, gold codes, predictions and scores to ``app/artifacts/showcase.json``
(gitignored: it contains real patient text).

    python app/build_showcase.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SHOWCASE, prefer_offline_hub  # noqa: E402

prefer_offline_hub()  # before transformers is imported
import inference as inf  # noqa: E402
from preprocessing.io_utils import load_jsonl  # noqa: E402

# Picked from a survey of the local test split (per-document F1 of this four-model app).
PICKS = [
    (6881, "Simple", "Short summary, 4 diagnoses, all found."),
    (8616, "Moderate", "8 diagnoses, several tied to explicit mentions in the text."),
    (3411, "Complex", "The most heavily coded test summary: 13 diagnoses."),
    (5414, "Hard", "Very long summary (8.5k characters); some diagnoses are missed."),
]


def main() -> None:
    by_id = {s["patient_id"]: s for s in load_jsonl(str(inf.REPO / "data/processed/test.jsonl"))}
    pipe = inf.CardioPipeline()
    out = []
    for pid, level, blurb in PICKS:
        s = by_id[pid]
        gold = s["document_level_annotations"]
        res = pipe.predict(s["text"])
        codes = set(res.final_codes) | {c for g in gold for c in g}
        out.append(
            {
                "patient_id": pid,
                "level": level,
                "blurb": blurb,
                "raw_text": s["text"],
                "gold": gold,
                "result": asdict(res),
                "score": inf.group_score(gold, res.final_codes, pipe.labelset),
                "code_desc": {c: pipe.code_desc.get(c, "") for c in sorted(codes)},
            }
        )
        print(f"#{pid} {level}: F1 {out[-1]['score']['micro_f1']:.2f}")
    SHOWCASE.parent.mkdir(parents=True, exist_ok=True)
    SHOWCASE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {SHOWCASE}")


if __name__ == "__main__":
    main()
