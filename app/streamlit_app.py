"""CardioICD demo: ICD-10 coding of cardiology discharge summaries.

    python app/run_demo.py          (recommended: models start loading immediately)
    streamlit run app/streamlit_app.py

The showcase documents are precomputed (``app/build_showcase.py``) and render instantly. The
models load in a background thread (``loader.py``), so they are usually ready by the time
someone writes their own note.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402  (light: no torch, renders instantly)

cm.prefer_offline_hub()
import loader  # noqa: E402
import render  # noqa: E402
from translate import is_mostly_greek  # noqa: E402

st.set_page_config(page_title="CardioICD", layout="wide")

EXAMPLE_EN = """\
A 72-year-old man was admitted through the emergency department with retrosternal chest pain lasting \
two hours, radiating to the left arm, together with sweating and nausea. He also reported progressive \
shortness of breath on exertion and swelling of both ankles over the previous two weeks.

Medical history: arterial hypertension for 15 years, type 2 diabetes mellitus on metformin, \
dyslipidaemia on a statin, and chronic kidney disease stage 3. Former smoker.

On admission the ECG showed atrial fibrillation with a rapid ventricular rate and ST-segment depression \
in the lateral leads. High-sensitivity troponin was elevated and rose on repeat measurement, consistent \
with a non-ST-elevation myocardial infarction. Chest X-ray showed bilateral pleural effusions and \
pulmonary congestion.

Transthoracic echocardiography revealed a dilated left ventricle with reduced ejection fraction (35%), \
hypokinesia of the anterior wall and moderate mitral regurgitation.

Coronary angiography showed a critical stenosis of the left anterior descending artery, which was \
treated with percutaneous coronary intervention and implantation of a drug-eluting stent, and \
non-significant lesions in the right coronary artery.

He received intravenous diuretics with clear improvement of the heart failure symptoms, and \
anticoagulation was started for the atrial fibrillation. He was discharged in good condition on dual \
antiplatelet therapy, an anticoagulant, a beta-blocker, an ACE inhibitor and a statin, with a cardiology \
follow-up in one month."""

BERT_SCORE_HELP = (
    "Greek BERT's probability for this code divided by its tuned threshold. 1.0 or more means "
    "Greek BERT on its own would predict the code; the final decision is made by the component "
    "shown under 'Decided by'."
)


# ---- data and models ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_showcase() -> list[dict]:
    import json

    return json.loads(cm.SHOWCASE.read_text(encoding="utf-8")) if cm.SHOWCASE.exists() else []


@st.cache_data(show_spinner=False)
def load_labelset() -> list[str]:
    from preprocessing.io_utils import LABELSET_PATH, load_labelset as _load

    return _load(LABELSET_PATH)


@st.cache_data(show_spinner=False)
def descriptions() -> tuple[dict[str, str], dict[str, str]]:
    return cm.load_english_descriptions(), cm.load_greek_descriptions()


def _model(name: str):
    """The loaded component, waiting for it if it is still warming up."""
    fut = loader.start()[name]
    if not fut.done():
        with st.spinner("The models are still warming up (only right after a fresh start)..."):
            fut.result()
    return fut.result()  # re-raises a loading error


@st.cache_data(max_entries=64, show_spinner=False)
def code_greek(greek: str) -> dict:
    return asdict(_model("pipeline").predict(greek))


@st.cache_data(max_entries=64, show_spinner=False)
def translate_pairs(text: str) -> list[tuple[str, str]]:
    return _model("translator").en_to_el_pairs(text)


def model_status() -> None:
    state = loader.status()
    if all(v == "ready" for v in state.values()):
        st.caption(":green[●] Models ready")
        return
    if "failed" in state.values():
        for name, fut in loader.start().items():
            if fut.done() and fut.exception() is not None:
                st.error(f"Loading the {name} failed: {fut.exception()}")
        return
    done = sum(v == "ready" for v in state.values())
    st.caption(
        f":orange[●] Loading the models in the background ({done}/{len(state)} ready, "
        f"{loader.seconds_since_start():.0f} s). You can already write your note."
    )


@st.fragment(run_every=2.0)
def model_status_live() -> None:
    """Polls while loading, then reruns the page once so the polling stops."""
    model_status()
    if all(v == "ready" for v in loader.status().values()):
        st.rerun()


# ---- results -----------------------------------------------------------------
def result_rows(res: cm.Result, gold: list[list[str]] | None, en: dict, el: dict) -> list[dict]:
    gold_codes = {c for g in gold for c in g} if gold is not None else set()
    rows = []
    for c in res.final_codes:
        status = "—" if gold is None else ("✅ correct" if c in gold_codes else "❌ extra")
        rows.append(
            {
                "ICD-10": c,
                "Description": en.get(c, ""),
                "Greek description": el.get(c, ""),
                "Status": status,
                "Decided by": cm.MODEL_LABELS.get(res.source[c], res.source[c]),
                "Greek BERT score": res.bert_scores.get(c, 0.0),
            }
        )
    if gold is not None:
        hit = set(res.final_codes)
        for g in gold:
            if not set(g) & hit:
                rows.append(
                    {
                        "ICD-10": " / ".join(g),
                        "Description": " / ".join(en.get(c, "") for c in g),
                        "Greek description": " / ".join(el.get(c, "") for c in g),
                        "Status": "⚠️ missed",
                        "Decided by": "—",
                        "Greek BERT score": max(res.bert_scores.get(c, 0.0) for c in g),
                    }
                )
    return rows


def _find_chunks(text: str, greek_chunks: list[str]) -> list[int] | None:
    """Start of each translated chunk inside the cleaned text that was scored."""
    from preprocessing.cleaning import clean_text

    starts, pos = [], 0
    for chunk in greek_chunks:
        probe = clean_text(chunk).strip()[:40]
        i = text.find(probe, pos) if probe else -1
        if i < 0:
            return None
        starts.append(i)
        pos = i + len(probe)
    return starts


def show_evidence(res: cm.Result, focus: set, tips: dict, pairs: list[tuple[str, str]] | None) -> None:
    spans = render.evidence_spans(res.mentions, res.final_codes)
    colour = render.colours(res.final_codes)
    st.markdown(
        f'<div style="margin-bottom:0.6rem">{render.legend_html(res.final_codes, spans, colour, tips, focus)}</div>',
        unsafe_allow_html=True,
    )
    starts = _find_chunks(res.text, [el for _, el in pairs]) if pairs else None
    if starts is None:
        body = render.text_html(res.text, spans, colour, tips, focus=focus, bert_chars=res.bert_chars)
        st.markdown(
            f'<div style="line-height:1.9;font-size:0.95rem;white-space:pre-wrap">{body}</div>',
            unsafe_allow_html=True,
        )
        return
    # English input: show each original piece next to the Greek that was coded.
    import html

    ends = starts[1:] + [len(res.text)]
    rows = "".join(
        f'<tr><td style="vertical-align:top;padding:6px 12px 6px 0;width:45%">{html.escape(en_txt)}</td>'
        f'<td style="vertical-align:top;padding:6px 0">'
        f"{render.text_html(res.text, spans, colour, tips, start=s, end=e, focus=focus, bert_chars=res.bert_chars)}"
        f"</td></tr>"
        for (en_txt, _), s, e in zip(pairs, starts, ends)
    )
    st.markdown(
        '<table style="width:100%;line-height:1.8;font-size:0.95rem;border-collapse:collapse">'
        '<tr><th style="text-align:left">Your text (English)</th><th style="text-align:left">'
        f"Greek translation that was coded</th></tr>{rows}</table>",
        unsafe_allow_html=True,
    )


def show_results(
    res: cm.Result,
    gold: list[list[str]] | None,
    key: str,
    pairs: list[tuple[str, str]] | None = None,
) -> None:
    en, el = descriptions()
    if gold is not None:
        score = cm.group_score(gold, res.final_codes, load_labelset())
        a, b, c, d = st.columns(4)
        a.metric("F1 (this document)", f"{score['micro_f1']:.2f}")
        b.metric("Precision", f"{score['precision']:.2f}")
        c.metric("Recall", f"{score['recall']:.2f}")
        d.metric("Codes predicted / expected", f"{len(res.final_codes)} / {len(gold)}")
        st.caption(
            "Scored with the official group-level rule: an expected group counts as found if any one "
            "of its codes is predicted."
        )
    else:
        st.metric("Codes predicted", len(res.final_codes))
        st.caption("Add the expected codes above to see precision, recall and F1.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("ICD-10 codes")
        rows = result_rows(res, gold, en, el)
        focus: set = set()
        if rows:
            event = st.dataframe(
                rows,
                hide_index=True,
                width="stretch",
                on_select="rerun",
                selection_mode="multi-row",
                key=f"codes-{key}",
                column_config={
                    "Greek description": st.column_config.TextColumn(width="medium"),
                    "Greek BERT score": st.column_config.ProgressColumn(
                        "Greek BERT score", help=BERT_SCORE_HELP, min_value=0.0, max_value=2.0, format="%.2f"
                    ),
                    "Decided by": st.column_config.TextColumn(
                        help="Each code is decided by the component that scored best on it during "
                        "validation (per-label routing)."
                    ),
                },
            )
            focus = {
                c for i in event.selection.rows for c in rows[i]["ICD-10"].split(" / ") if c in res.final_codes
            }
            st.caption("Select rows to focus the evidence below on those codes.")
        else:
            st.info("No codes were predicted for this text.")
    with right:
        st.subheader("What each model said")
        for name, label in cm.MODEL_LABELS.items():
            st.markdown(f"**{label}**: " + (", ".join(res.per_model[name]) or "none"))
        st.caption(
            "The final codes combine these: each code is decided by the component that scored best "
            "on it during validation."
        )

    st.subheader("Evidence in the text")
    note = (
        "Each predicted code has its own colour; hover a highlight for its description. Spans come "
        "from dictionary term matches and the NER + entity-linking model. Codes struck through in the "
        "legend have no single phrase to point at (usually Greek BERT, which judges the text as a whole)."
    )
    if res.bert_chars is not None:
        note += (
            " This document is longer than Greek BERT's window (256 word pieces), so the shaded part "
            "after the marker was only seen by the other components."
        )
    st.caption(note)
    tips = {c: render.describe(c, en, el) for c in res.final_codes}
    show_evidence(res, focus, tips, pairs)


# ---- page --------------------------------------------------------------------
st.title("CardioICD")
st.write(
    "Automatic ICD-10 coding of Greek cardiology discharge summaries. "
    "Pick a held-out example and see how the system scores, or type your own note."
)
st.caption(
    "Research demo, not for clinical use. It runs a lighter version of our system, "
    "so it is somewhat less accurate than the full model."
)

missing = cm.missing_artifacts()
if not missing:
    loader.start()  # no-op if run_demo.py already started it

mode = st.radio("Input", ["Showcase examples", "Write your own"], horizontal=True)

if mode == "Showcase examples":
    showcase = load_showcase()
    if not showcase:
        st.warning(
            "The showcase file (`app/artifacts/showcase.json`) is not on this machine. It holds real "
            "patient text and is never committed; build it with `python app/build_showcase.py`, or use "
            "**Write your own**."
        )
        st.stop()
    showcase = sorted(showcase, key=lambda s: s["patient_id"])
    names = [f"Discharge summary #{s['patient_id']}" for s in showcase]
    pick = st.selectbox("Held-out test document", names)
    sample = showcase[names.index(pick)]
    with st.expander("Show the discharge summary (Greek)"):
        st.write(sample["raw_text"])
    show_results(cm.Result.from_dict(sample["result"]), sample["gold"], key=f"show-{sample['patient_id']}")
else:
    if missing:
        st.error(
            "Writing your own note needs the trained models and data, and some files are missing:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
        )
        st.stop()
    if all(v == "ready" for v in loader.status().values()):
        model_status()
    else:
        model_status_live()

    en, _ = descriptions()
    text = st.text_area("Discharge summary (Greek or English)", height=360, value=EXAMPLE_EN, key="note")
    expected = st.multiselect(
        "Optional: expected ICD-10 codes (to compute F1)",
        load_labelset(),
        format_func=lambda c: f"{c} — {en[c]}" if c in en else c,
        help="Leave empty to just see predictions. You can change these after coding; the scores update.",
    )
    if st.button("Code this note", type="primary", disabled=not text.strip()):
        try:
            pairs = None
            greek = text
            if not is_mostly_greek(text):
                with st.spinner("Translating English to Greek..."):
                    pairs = translate_pairs(text)
                greek = " ".join(el for _, el in pairs)
            with st.spinner("Coding..."):
                result = code_greek(greek)
            st.session_state["last"] = {"text": text, "result": result, "pairs": pairs}
        except Exception as exc:  # loading or inference error: show it instead of a stack trace
            st.error(f"Could not code this note: {exc}")

    last = st.session_state.get("last")
    if last:
        if last["text"] != text:
            st.info("The text has changed. These results are for the previous version; press **Code this note** to update.")
        show_results(
            cm.Result.from_dict(last["result"]),
            [[c] for c in expected] if expected else None,
            key="own",
            pairs=last["pairs"],
        )
