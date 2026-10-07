"""Offline English -> Greek translation (nothing leaves the machine)."""
from __future__ import annotations

import os
import re
from typing import List, Tuple

MODEL = "Helsinki-NLP/opus-mt-tc-big-en-el"


def _chunks(text: str, max_words: int = 60) -> List[str]:
    """Split into sentence-sized pieces so long notes translate fully (MT models truncate)."""
    parts = re.split(r"(?<=[\.\!\?;])\s+|\n+", text.strip())
    out, cur = [], []
    for p in parts:
        w = p.split()
        if cur and len(cur) + len(w) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur += w
    if cur:
        out.append(" ".join(cur))
    return [c for c in out if c.strip()]


class Translator:
    """English -> Greek with Helsinki-NLP opus-mt-tc-big-en-el.

    Uses CTranslate2 when it is installed: the same model and beam search, ~4-5x faster than
    ``transformers.generate`` (whose per-step overhead dominates for this model). The CTranslate2
    copy is converted once and kept under ``outputs/models/``.
    """

    def __init__(self, device: str | None = None) -> None:
        import threading

        import torch
        from transformers import MarianTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = MarianTokenizer.from_pretrained(MODEL)
        self._lock = threading.Lock()
        self.ct2 = self._load_ct2()
        self.model = None
        if self.ct2 is None:
            from transformers import MarianMTModel

            self.model = MarianMTModel.from_pretrained(MODEL).to(self.device).eval()

    def _load_ct2(self):
        try:
            import ctranslate2
        except ImportError:
            return None
        from common import CT2_DIR

        if not (CT2_DIR / "model.bin").exists():
            from ctranslate2.converters import TransformersConverter

            TransformersConverter(MODEL).convert(str(CT2_DIR), quantization="float16", force=True)
        # int8 weights with float32 compute: ~4x less GPU memory than float32, which matters when
        # the translator shares a small GPU with the coding models (on a 4 GB card the last model
        # loaded otherwise spills into system memory and runs ~4x slower). float16 is avoided: GPUs
        # without tensor cores (GTX 16xx) run it ~3x slower. Override with CARDIO_CT2_COMPUTE.
        default = "int8_float32" if self.device.type == "cuda" else "int8"
        compute = os.environ.get("CARDIO_CT2_COMPUTE", default)
        return ctranslate2.Translator(str(CT2_DIR), device=self.device.type, compute_type=compute)

    def _translate(self, pieces: List[str], batch_size: int = 8) -> List[str]:
        if self.ct2 is not None:
            toks = [self.tok.convert_ids_to_tokens(self.tok.encode(p)) for p in pieces]
            res = self.ct2.translate_batch(toks, beam_size=4, max_decoding_length=256, max_batch_size=batch_size)
            return [self.tok.decode(self.tok.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True) for r in res]
        out: List[str] = []
        for i in range(0, len(pieces), batch_size):
            enc = self.tok(pieces[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256)
            gen = self.model.generate(**{k: v.to(self.device) for k, v in enc.items()}, num_beams=4, max_new_tokens=256)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return out

    def en_to_el_pairs(self, text: str) -> List[Tuple[str, str]]:
        """``(English piece, Greek translation)`` per sentence-sized piece, in order."""
        pieces = _chunks(text)
        with self._lock:
            return list(zip(pieces, self._translate(pieces)))
