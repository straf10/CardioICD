"""Live CardioICD inference for a single Greek discharge summary.

Runs the same components, with the same settings, that produced the saved
prediction files, then fuses them with the per-label routing strategy
(each ICD code is decided by its validation-champion component).

Components: Greek BERT (multi-label), dictionary baseline, hybrid IR, NER+EL.
"""
from __future__ import annotations

import json
import pickle
import threading
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

import numpy as np
import torch
import yaml

from common import (  # noqa: F401  (re-exported for callers)
    GBERT_BASE,
    GBERT_CFG,
    GBERT_DIR,
    IR_CACHE,
    MODEL_LABELS,
    NER_DIR,
    REPO,
    ROUTING,
    Mention,
    Result,
    group_score,
    missing_artifacts,
)
from evidence import build_evidence_automaton, find_evidence
from dictionary.config import get_config_path_default, load_dictionary_config
from dictionary.export import load_code_description_csv
from dictionary.matcher import build_automaton, load_term_code_csv, predict_codes_for_text
from mlc_greek_bert.model import MLCModel
from preprocessing.cleaning import clean_text
from preprocessing.io_utils import LABELSET_PATH, load_jsonl, load_labelset

IR_EMBEDDER = "intfloat/multilingual-e5-base"
IR_TOKENIZER_CACHE = IR_CACHE.with_name("e5_tokenizer.pkl")


def strip_accents_and_lowercase(text: str) -> str:
    """Same as ``mlc_greek_bert.train.strip_accents_and_lowercase``; copied because importing
    that training module pulls in wandb (~2 s of start-up)."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn").lower()


def _stripped_with_offsets(text: str) -> tuple[str, list[int]] | None:
    """``strip_accents_and_lowercase(text)`` plus, per character, its index in ``text``."""
    chars, src = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFD", ch):
            if unicodedata.category(c) != "Mn":
                chars.append(c)
                src.append(i)
    out = "".join(chars).lower()
    return (out, src) if out == strip_accents_and_lowercase(text) else None


class CardioPipeline:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.labelset = load_labelset(LABELSET_PATH)
        route = json.loads(ROUTING.read_text(encoding="utf-8"))
        self.routing: Dict[str, str] = route["routing"]
        self.cutoff: float = float(route["score_cutoff"])
        # Several browser sessions share one pipeline; run one prediction at a time.
        self._lock = threading.Lock()
        self._load_bert()
        self._load_dictionary()
        self._load_ner()
        self._load_ir()

    # ---- component loading -------------------------------------------------
    def _load_bert(self) -> None:
        from transformers import AutoTokenizer

        cfg = yaml.safe_load(GBERT_CFG.read_text(encoding="utf-8"))
        m = cfg["model"]
        self.bert_tok = AutoTokenizer.from_pretrained(str(GBERT_BASE))
        self.bert_len = int(m["max_length"])
        self.bert_labels = json.loads((GBERT_DIR / "labels.json").read_text(encoding="utf-8"))
        thr = json.loads((GBERT_DIR / "best_thresholds.json").read_text(encoding="utf-8"))
        thr = thr.get("thresholds", thr)
        self.bert_thr = np.array([float(thr[l]) for l in self.bert_labels], dtype=np.float32)
        # The checkpoint holds every weight, so build the encoder from its config straight on the
        # target device instead of first reading the pretrained base weights (~0.3 s vs ~3 s).
        from transformers import AutoConfig, AutoModel

        hf_cfg = AutoConfig.from_pretrained(str(GBERT_BASE))
        from_config = mock.patch.object(AutoModel, "from_pretrained", lambda _n, **kw: AutoModel.from_config(hf_cfg, **kw))
        with from_config, self.device:
            model = MLCModel(
                model_name=str(GBERT_BASE),
                num_labels=int(m["num_labels"]),
                dropout=0.0,
                pooling=str(m.get("pooling", "cls")),
                head=str(m.get("head", "linear")),
                head_hidden_dim=m.get("head_hidden_dim"),
            )
        state = torch.load(GBERT_DIR / "best_model.pt", map_location=self.device, weights_only=True, mmap=True)
        model.load_state_dict(state)
        self.bert = model.eval()

    def _load_dictionary(self) -> None:
        self.dict_cfg = load_dictionary_config(get_config_path_default())
        term_map = load_term_code_csv(self.dict_cfg.paths["term_code_csv"], blacklist=self.dict_cfg.blacklist)
        wb = bool((self.dict_cfg.matching or {}).get("word_boundary", False))
        self.dict_matcher = build_automaton(term_map, word_boundary=wb)
        self.code_desc = load_code_description_csv(self.dict_cfg.paths["code_description_csv"])
        self.dict_term_map = term_map
        self.evidence = build_evidence_automaton(term_map)

    def _load_ner(self) -> None:
        from ner_el.service import NERELService

        self.ner = NERELService.from_model_dir(str(NER_DIR))

    def _load_ir(self) -> None:
        from dictionary.dictionary import build_automaton as ir_build
        from information_retrieval.prediction import IRPredictionParams

        self.ir = self._fitted_ir()
        # Tuned on validation by ``information_retrieval.predict --tune`` (see run log).
        self.ir_params = IRPredictionParams(
            search_top_k=80, fraction_of_top_score=0.04, max_codes=2, min_ir_codes=0, include_dictionary=True
        )
        # Same CSV and normalisation as the dictionary component, so reuse its term map.
        self.ir_term_map = ir_build(self.dict_term_map)

    def _fitted_ir(self):
        """Hybrid retriever; the fitted index (BM25 + e5 doc embeddings) is cached on disk
        because embedding the corpus dominates start-up time (~13 s)."""
        from information_retrieval.corpus import build_code_documents_with_mention_expansion
        from information_retrieval.evaluate import fit_retriever
        from preprocessing.io_utils import RAW_TRAIN_PATH, RAW_VAL_PATH

        raw = [Path(p) for p in (RAW_TRAIN_PATH, RAW_VAL_PATH)]
        have_raw = all(p.exists() for p in raw)
        key = [IR_EMBEDDER, *(p.stat().st_mtime_ns for p in raw), len(self.labelset)] if have_raw else None
        if IR_CACHE.exists():
            with IR_CACHE.open("rb") as f:
                cached = pickle.load(f)
            # Without the raw data the cache can't be checked for staleness, but it is all we have.
            if not have_raw or cached["key"] == key:
                ir = cached["ir"]
                ir._dense._model = self._embedder()
                return ir

        docs = build_code_documents_with_mention_expansion(
            load_jsonl(str(RAW_TRAIN_PATH)) + load_jsonl(str(RAW_VAL_PATH)), codes=self.labelset
        )
        ir = fit_retriever(
            "hybrid",
            docs,
            embedding_model=IR_EMBEDDER,
            hybrid_rrf_k=30,
            hybrid_bm25_weight=1.0,
            hybrid_dense_weight=0.4,
        )
        model, ir._dense._model = ir._dense._model, None
        IR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with IR_CACHE.open("wb") as f:
            pickle.dump({"key": key, "ir": ir}, f)
        ir._dense._model = model
        return ir

    def _embedder(self):
        """e5 SentenceTransformer. Its 17 MB tokenizer takes ~2.6 s to parse with transformers 5,
        so the built tokenizer is pickled once and reused (~0.6 s)."""
        from information_retrieval.embedding_retrieval import _require_sentence_transformers

        SentenceTransformer = _require_sentence_transformers()
        try:
            import sentence_transformers.base.modules.transformer as st_tf

            real = st_tf.AutoProcessor
        except (ImportError, AttributeError):  # other sentence-transformers layout: plain load
            return SentenceTransformer(IR_EMBEDDER, device=str(self.device))

        class _CachedProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                if IR_TOKENIZER_CACHE.exists():
                    return pickle.loads(IR_TOKENIZER_CACHE.read_bytes())
                proc = real.from_pretrained(*args, **kwargs)
                IR_TOKENIZER_CACHE.parent.mkdir(parents=True, exist_ok=True)
                IR_TOKENIZER_CACHE.write_bytes(pickle.dumps(proc))
                return proc

        with mock.patch.object(st_tf, "AutoProcessor", _CachedProcessor):
            return SentenceTransformer(IR_EMBEDDER, device=str(self.device))

    # ---- per-component prediction -----------------------------------------
    @torch.inference_mode()
    def _bert_scores(self, text: str) -> tuple[np.ndarray, Optional[int]]:
        """Probabilities, and how many characters of ``text`` fit in BERT's window
        (None when the whole text fits)."""
        enc = self.bert_tok(
            strip_accents_and_lowercase(text),
            add_special_tokens=True,
            max_length=self.bert_len,
            truncation=True,
            padding=False,
            return_token_type_ids=False,
        )
        x = torch.tensor([enc["input_ids"]], dtype=torch.long, device=self.device)
        logits = self.bert(input_ids=x, attention_mask=torch.ones_like(x))
        return torch.sigmoid(logits.float()).cpu().numpy()[0], self._bert_window_chars(text, len(enc["input_ids"]))

    def _bert_window_chars(self, text: str, n_ids: int) -> Optional[int]:
        if n_ids < self.bert_len:
            return None
        mapped = _stripped_with_offsets(text)
        if mapped is None:
            return None
        stripped, src = mapped
        offsets = self.bert_tok(stripped, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if len(offsets) <= self.bert_len - 2:  # [CLS] and [SEP] take two slots
            return None
        end = offsets[self.bert_len - 3][1]  # end of the last word piece that fits
        return src[end - 1] + 1 if end > 0 else 0

    def _ir_codes(self, text: str) -> List[str]:
        from information_retrieval.evaluate import predict_ir_codes_for_records

        out = predict_ir_codes_for_records(
            [{"patient_id": 1, "text": text}],
            self.ir,
            params=self.ir_params,
            term_code_map=self.ir_term_map,
            strategy="standard",
            fallback_to_standard_if_no_dictionary=True,
        )
        return list(out[1])

    # ---- full pipeline ----------------------------------------------------
    def predict(self, greek_text: str) -> Result:
        with self._lock:
            return self._predict(greek_text)

    def _predict(self, greek_text: str) -> Result:
        text = clean_text(greek_text)
        labels = self.labelset

        probs, bert_chars = self._bert_scores(text)
        score = {l: float(p / t) if t > 0 else 0.0 for l, p, t in zip(self.bert_labels, probs, self.bert_thr)}
        per_model: Dict[str, List[str]] = {
            "mlc_greek_bert": [l for l in labels if score.get(l, 0.0) >= self.cutoff],
            "dictionary_baseline": sorted(
                c
                for c in predict_codes_for_text(
                    text, self.dict_matcher, config=self.dict_cfg, labelset=labels, code_desc_map=self.code_desc
                )
                if c in set(labels)
            ),
            "information_retrieval": [c for c in self._ir_codes(text) if c in set(labels)],
        }
        out = self.ner.predict_text(0, text)
        ner_codes = [g[0] for g in out.doc_prediction["document_level_annotations"] if g]
        per_model["ner_el"] = [c for c in ner_codes if c in set(labels)]
        fired = {n: set(v) for n, v in per_model.items()}
        final, source = [], {}
        for l in labels:
            champ = self.routing.get(l, "mlc_greek_bert")
            if l in fired.get(champ, set()):
                final.append(l)
                source[l] = champ

        # Evidence only for codes the system actually predicted.
        kept = set(final)
        mentions = [
            Mention(s, e, text[s:e], c, 1.0, "dictionary") for s, e, c in find_evidence(text, self.evidence, kept)
        ] + [
            Mention(m["start"], m["end"], m["mention"], m["code"], float(m.get("confidence") or 0.0), "ner")
            for m in out.debug_prediction["mention_level_annotations"]
            if m["code"] in kept
        ]
        return Result(text, final, source, per_model, score, mentions, bert_chars)

