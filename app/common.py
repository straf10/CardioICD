"""Light shared definitions for the app (no torch / transformers imports, so the UI renders instantly)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

GBERT_DIR = REPO / "outputs/models/mlc_greek_bert/p4_winner_lean_head"
GBERT_BASE = REPO / "outputs/models/greek_bert_base"
GBERT_CFG = REPO / "src/mlc_greek_bert/mlc_greek_bert.yaml"
NER_DIR = REPO / "outputs/models/ner_el"
ROUTING = Path(__file__).resolve().parent / "artifacts/routing.json"
# Built from real patient text, so both stay local (gitignored) like the data itself.
IR_CACHE = REPO / "outputs/models/ir_app_cache/hybrid_e5.pkl"
SHOWCASE = Path(__file__).resolve().parent / "artifacts/showcase.json"
# CTranslate2 copy of the public EN->EL model (converted on first use; large, so not in git).
CT2_DIR = REPO / "outputs/models/ct2_opus_en_el"

MODEL_LABELS = {
    "mlc_greek_bert": "Greek BERT",
    "dictionary_baseline": "Dictionary",
    "information_retrieval": "Retrieval (IR)",
    "ner_el": "NER + entity linking",
}


@dataclass
class Mention:
    start: int
    end: int
    text: str
    code: str
    confidence: float
    via: str = "ner"  # "dictionary" (term match) or "ner" (NER + entity linking)


@dataclass
class Result:
    text: str  # cleaned Greek text that was actually scored
    final_codes: List[str]
    source: Dict[str, str]  # code -> champion model that decided it
    per_model: Dict[str, List[str]]
    bert_scores: Dict[str, float]  # code -> prob / threshold (>=1 means positive)
    mentions: List[Mention] = field(default_factory=list)
    # Greek BERT reads at most 256 word pieces: characters of ``text`` it saw (None = all of it).
    bert_chars: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        return cls(**{**d, "mentions": [Mention(**m) for m in d.get("mentions", [])]})


HUB_MODELS = ["intfloat/multilingual-e5-base", "Helsinki-NLP/opus-mt-tc-big-en-el"]


def prefer_offline_hub() -> bool:
    """Skip Hugging Face Hub update checks (~6 s of start-up) when every hub model is already
    cached locally. Must run before ``huggingface_hub`` is imported, which reads the flag once."""
    if "HF_HUB_OFFLINE" in os.environ:
        return os.environ["HF_HUB_OFFLINE"] not in ("0", "false", "False", "")
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface"), "hub"
    )
    cached = all(
        any((snap / "config.json").exists() for snap in (Path(hub) / f"models--{m.replace('/', '--')}" / "snapshots").glob("*"))
        for m in HUB_MODELS
    )
    if cached:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return cached


def missing_artifacts() -> List[str]:
    need = [
        GBERT_DIR / "best_model.pt",
        GBERT_DIR / "best_thresholds.json",
        GBERT_DIR / "labels.json",
        GBERT_BASE / "model.safetensors",
        NER_DIR / "model.safetensors",
        NER_DIR / "partial_crf.pt",
        ROUTING,
    ]
    if not IR_CACHE.exists():  # the retrieval index is built from the training data once
        need += [REPO / "data/raw/train.jsonl", REPO / "data/raw/val.jsonl"]
    return [str(p.relative_to(REPO)) for p in need if not p.exists()]


def group_score(gold_groups: List[List[str]], codes: List[str], labelset: List[str]) -> dict:
    """Official group-level micro P/R/F1 for one document (via evaluation.evaluator, no torch)."""
    from evaluation.evaluator import evaluate_data

    m = evaluate_data({0: gold_groups}, {0: list(codes)}, label_space=labelset)
    return {k: float(m[k]) for k in ("micro_f1", "precision", "recall")} | {
        k: int(m[k]) for k in ("total_tp", "total_fp", "total_fn") if k in m
    }


EN_DESC_CSV = REPO / "data/external/icd10_english_lookup.csv"
EL_DESC_CSV = REPO / "data/external/icd10_greek_lookup.csv"


def load_greek_descriptions() -> Dict[str, str]:
    """Greek ICD-10 descriptions (the same file the dictionary component uses)."""
    import csv

    with EL_DESC_CSV.open(encoding="utf-8", newline="") as f:
        return {r["code"].strip(): r["greek_description"].strip() for r in csv.DictReader(f)}


def load_english_descriptions() -> Dict[str, str]:
    """WHO ICD-10 English titles for the 115 target codes (code -> title)."""
    import csv

    if not EN_DESC_CSV.exists():
        return {}
    with EN_DESC_CSV.open(encoding="utf-8", newline="") as f:
        return {r["code"]: r["title_en"] for r in csv.DictReader(f)}
