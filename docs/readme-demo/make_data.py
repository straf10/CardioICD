"""Build data.js for the README demo scenes (manual run).

Run from the repository root:  python -I docs/readme-demo/make_data.py

Local rerun values (threshold curve) come from the Greek BERT validation scores and the gold labels in
data/raw/val.jsonl (not tracked). Only aggregates are written. Paper values are copied from the paper
(report/main.pdf) and the README tables.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.io_utils import load_ground_truth  # noqa: E402
from evaluation.scoring import micro_f1, score_document  # noqa: E402

M = ROOT / "outputs/models/mlc_greek_bert/p4_winner_lean_head"
scores = np.load(M / "val_scores.npy")
labels = json.loads((M / "labels.json").read_text(encoding="utf-8"))
pids = json.loads((M / "val_pids.json").read_text(encoding="utf-8"))
tuned = json.loads((M / "best_thresholds.json").read_text(encoding="utf-8"))
gold = load_ground_truth(str(ROOT / "data/raw/val.jsonl"))


def f1_at(thr):
    tp = fp = fn = 0
    for i, p in enumerate(pids):
        pred = [labels[j] for j in range(len(labels)) if scores[i, j] >= thr[j]]
        a, b, c = score_document(gold[int(p)], pred)
        tp, fp, fn = tp + a, fp + b, fn + c
    return micro_f1(tp, fp, fn)[2]


xs = [round(0.45 + 0.01 * k, 2) for k in range(31)]
curve = [round(f1_at([x] * len(labels)), 4) for x in xs]
per_label = [tuned["thresholds"][c] for c in labels]
f1_tuned = f1_at(per_label)
assert abs(f1_tuned - tuned["best_micro_f1"]) < 1e-6, "per-label F1 differs from the saved tuning run"
best_global = max(range(len(xs)), key=curve.__getitem__)

# Synthetic demo note (docs/demo-data.js). Spans are the highlighted parts of text_html.
demo = json.loads((ROOT / "docs/demo-data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
ex = next(e for e in demo["examples"] if e["id"] == "syn-3")
spans, plain, pos = [], "", 0
for m in re.finditer(r'<mark[^>]*data-codes="([^"]+)"[^>]*>(.*?)<sup[^>]*>.*?</sup></mark>', ex["text_html"]):
    plain += re.sub(r"<[^>]+>", "", ex["text_html"][pos:m.start()])
    spans.append([len(plain), len(plain) + len(m.group(2)), m.group(1)])
    plain += m.group(2)
    pos = m.end()
plain += re.sub(r"<[^>]+>", "", ex["text_html"][pos:])

data = {
    "thr": {"x": xs, "f1": curve, "at05": curve[xs.index(0.5)], "best_x": xs[best_global],
            "best_f1": curve[best_global], "per_label": per_label, "labels": labels, "tuned": round(f1_tuned, 4)},
    # Paper Table 5 (validation micro-F1 per component)
    "components": [["Greek BERT", 0.8165], ["XLM-R Base", 0.8101], ["XLM-R Large", 0.76],
                   ["NER + EL", 0.7223], ["Dictionary", 0.7176], ["Retrieval", 0.6621]],
    # README Results table (official test set)
    "test": [["Ensemble merge_and", 0.8510, 0.8830, 0.8667], ["Ensemble merge_k2", 0.8687, 0.8539, 0.8613],
             ["Ensemble safer threshold", 0.8767, 0.8431, 0.8596], ["Greek BERT 100% data", 0.8194, 0.8811, 0.8491],
             ["Greek BERT 80/10/10", 0.8310, 0.8675, 0.8489]],
    "boot": {"mean": 0.0323, "lo": 0.0212, "hi": 0.0437, "n": 10000},
    "mcnemar": {"ens": 88, "bert": 21},
    "bands": {"rare": {"Dictionary": 0.62, "NER + EL": 0.62, "Greek BERT": 0.37, "XLM-R Large": 0.35, "Ensemble": 0.39},
              "freq": {"Greek BERT": 0.92, "Dictionary": 0.71}},
    "sizes": {"docs": 2500, "codes": 115, "train": 2000, "val": 250, "test": 250},
    "cost": [["XLM-R Large", "560M", "2h 14m"], ["XLM-R Base", "278M", "1h 36m"], ["Greek BERT", "113M", "15m"],
             ["NER + EL", "113M", "3m"], ["Retrieval", "-", "30s"], ["Dictionary", "-", "5s"]],
    "note": {"text": plain, "spans": spans, "rows": [[r["code"], r["decided_by"], r["bert"]] for r in ex["rows"]
                                                     if r["status"] == "correct"], "f1": round(ex["score"]["micro_f1"], 3)},
}
(Path(__file__).parent / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
sys.stdout.write(f"at0.5={curve[xs.index(0.5)]} best={xs[best_global]}:{curve[best_global]} tuned={f1_tuned:.4f} spans={len(spans)}\n")
