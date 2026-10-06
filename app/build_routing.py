"""Build the per-label routing table the app uses (champion model per ICD code).

Re-derives, from the saved validation predictions, exactly what
``python -m ensemble_metaheuristic`` computes for its ``per_label_routing`` strategy,
and writes a small label-level JSON (no patient text) to ``app/artifacts/routing.json``.

    PYTHONPATH=src python app/build_routing.py --config src/evaluation/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model
from ensemble_metaheuristic.strategies.per_label_routing import (
    build_label_routing_table,
    per_label_f1,
)
from ensemble_metaheuristic.strategy_loaders import (
    canonical_ensemble_label_arts,
    gather_ensemble_artifacts,
)
from evaluation.config_utils import get_cfg, load_config
from evaluation.io_utils import load_ground_truth

OUT = Path(__file__).resolve().parent / "artifacts" / "routing.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/evaluation/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt = load_ground_truth(str(get_cfg(cfg, "data.val_path")))
    pids = list(gt.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}
    arts = gather_ensemble_artifacts(model_cfgs, pids, "val")
    labels = canonical_ensemble_label_arts(arts).label_names

    names, preds = [], {}
    for name, a in arts:
        thr = load_thresholds_for_model(model_cfgs[name], labels) if a.scores is not None else None
        mat = build_score_matrix(a, pids, labels, thr)
        cut = 1.0 if a.scores is not None else 0.5
        preds[name] = {pid: [labels[j] for j in np.where(mat[i] >= cut)[0]] for i, pid in enumerate(pids)}
        names.append(name)

    f1s = {n: per_label_f1(gt, preds[n], labels) for n in names}
    routing = build_label_routing_table(f1s, labels)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "models": names,
                "score_cutoff": 1.0,  # best cutoff found by the ensemble sweep
                "routing": routing,
                "val_label_f1": {n: {l: round(f1s[n].get(l, 0.0), 4) for l in labels} for n in names},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    counts = {n: sum(1 for v in routing.values() if v == n) for n in names}
    print("wrote", OUT, "| labels per champion:", counts)


if __name__ == "__main__":
    main()
