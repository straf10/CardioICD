"""Check the live pipeline reproduces the saved prediction files on test documents.

    PYTHONPATH=src python app/check_parity.py --n 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import CardioPipeline, REPO  # noqa: E402

from preprocessing.io_utils import load_jsonl  # noqa: E402

SAVED = {
    "mlc_greek_bert": "outputs/predictions/mlc_greek_bert/test_predictions.jsonl",
    "dictionary_baseline": "outputs/predictions/dictionary_baseline/test_predictions.jsonl",
    "information_retrieval": "outputs/predictions/information_retrieval/test_predictions.jsonl",
    "ner_el": "outputs/predictions/ner_el/test_predictions.jsonl",
    "FINAL (per_label_routing)": "outputs/predictions/ensemble_metaheuristic/per_label_routing/test_predictions.jsonl",
}


def load(p):
    out = {}
    for line in open(REPO / p, encoding="utf-8"):
        r = json.loads(line)
        out[int(r["patient_id"])] = {c for g in r["document_level_annotations"] for c in (g if isinstance(g, list) else [g])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    n = ap.parse_args().n
    t = time.time()
    pipe = CardioPipeline()
    print(f"loaded in {time.time()-t:.0f}s")
    saved = {k: load(v) for k, v in SAVED.items()}
    docs = load_jsonl(str(REPO / "data/processed/test.jsonl"))[:n]
    same = {k: 0 for k in SAVED}
    t = time.time()
    for d in docs:
        pid = int(d["patient_id"])
        r = pipe.predict(d["text"])
        got = {**{k: set(v) for k, v in r.per_model.items()}, "FINAL (per_label_routing)": set(r.final_codes)}
        for k in SAVED:
            if got[k] == saved[k].get(pid):
                same[k] += 1
            else:
                print(f"  diff {k} pid={pid}: live-only={sorted(got[k]-saved[k].get(pid,set()))} saved-only={sorted(saved[k].get(pid,set())-got[k])}")
    print(f"{(time.time()-t)/len(docs):.2f}s/doc")
    for k, v in same.items():
        print(f"{k:28s} identical on {v}/{len(docs)} docs")


if __name__ == "__main__":
    main()
