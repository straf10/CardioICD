"""Background model loading, shared by the whole server process.

A plain module (not ``st.cache_resource``) so that ``run_demo.py`` can start loading before any
browser connects: Streamlit only runs the page script once a session opens, but imported modules
live for the whole process, so the page later picks up the same futures.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional

import common as cm

_lock = threading.Lock()
_futures: Optional[Dict[str, Future]] = None
_started_at: Optional[float] = None

# Small notes used to pre-warm GPU kernels and caches, so the first real request is fast.
_WARM_EL = "Ασθενής με κολπική μαρμαρυγή, αρτηριακή υπέρταση και καρδιακή ανεπάρκεια."
_WARM_EN = "The patient has atrial fibrillation and hypertension."


def _build_pipeline():
    import inference as inf

    pipe = inf.CardioPipeline()
    pipe.predict(_WARM_EL)
    return pipe


def _build_translator():
    from translate import Translator

    tr = Translator()
    tr.en_to_el(_WARM_EN)
    return tr


def start() -> Dict[str, Future]:
    """Start loading once per process (idempotent). Pipeline first, then the translator."""
    global _futures, _started_at
    with _lock:
        if _futures is None:
            cm.prefer_offline_hub()
            _started_at = time.time()
            # One worker: transformers model loading is not thread-safe (lazy imports and a
            # process-wide meta-device context), so components load one after another.
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cardio-load")
            _futures = {"pipeline": pool.submit(_build_pipeline), "translator": pool.submit(_build_translator)}
        return _futures


def status() -> Dict[str, str]:
    """``loading`` / ``ready`` / ``failed`` per component."""
    out = {}
    for name, fut in start().items():
        if not fut.done():
            out[name] = "loading"
        else:
            out[name] = "failed" if fut.exception() is not None else "ready"
    return out


def seconds_since_start() -> float:
    return 0.0 if _started_at is None else time.time() - _started_at
