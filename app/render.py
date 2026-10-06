"""HTML for the evidence view, shared by the Streamlit app and the static GitHub Pages export.

Every highlight carries ``data-codes`` so the static page can focus codes with plain JS, and the
Streamlit app focuses them by re-rendering with ``focus``.
"""
from __future__ import annotations

import html
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Light backgrounds with dark text read well on both light and dark themes.
PALETTE = ["#ffe08a", "#b9e4c9", "#bcd7ff", "#ffc9b9", "#e2c9ff", "#c4f0ef", "#ffd1e8", "#dfe7a8", "#f5d3a7", "#d0d4dc"]

Span = Tuple[int, int, List[str]]


def colours(codes: Sequence[str]) -> Dict[str, str]:
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(codes)}


def describe(code: str, desc_en: Dict[str, str], desc_el: Dict[str, str]) -> str:
    parts = [p for p in (desc_en.get(code, ""), desc_el.get(code, "")) if p]
    return f"{code}: " + " / ".join(parts) if parts else code


def evidence_spans(mentions: Iterable, codes: Sequence[str]) -> List[Span]:
    """Non-overlapping ``(start, end, codes)``; when spans overlap the longest wins."""
    wanted = set(codes)
    by_span: Dict[Tuple[int, int], List[str]] = {}
    for m in mentions:
        if m.end > m.start and m.code in wanted:
            cs = by_span.setdefault((m.start, m.end), [])
            if m.code not in cs:
                cs.append(m.code)
    spans, last = [], 0
    for (s, e), cs in sorted(by_span.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        if s >= last:
            spans.append((s, e, cs))
            last = e
    return spans


def _mark(text: str, s: int, e: int, cs: List[str], colour: Dict[str, str], tip: str, dim: bool) -> str:
    style = f"background:{colour[cs[0]]};color:#222;padding:1px 3px;border-radius:3px"
    if dim:
        style += ";opacity:0.25"
    return (
        f'<mark class="ev" data-codes="{html.escape(" ".join(cs), quote=True)}" title="{html.escape(tip, quote=True)}" '
        f'style="{style}">{html.escape(text[s:e])}'
        f'<sup style="font-weight:600;margin-left:3px">{html.escape(", ".join(cs))}</sup></mark>'
    )


def text_html(
    text: str,
    spans: List[Span],
    colour: Dict[str, str],
    tips: Dict[str, str],
    *,
    start: int = 0,
    end: Optional[int] = None,
    focus: Optional[set] = None,
    bert_chars: Optional[int] = None,
) -> str:
    """``text[start:end]`` with its evidence spans highlighted. Text Greek BERT never read
    (after ``bert_chars``) is shaded, with a marker where its window ends."""
    end = len(text) if end is None else end
    cut = bert_chars if bert_chars is not None and start <= bert_chars < end else None
    out, pos, cut_done = [], start, cut is None

    def plain(a: int, b: int) -> None:
        nonlocal cut_done
        if not cut_done and a <= cut <= b:
            out.append(html.escape(text[a:cut]))
            out.append(
                '<span class="bert-cut" title="Greek BERT reads at most 256 word pieces; the text after '
                'this point is only seen by the dictionary, retrieval and NER + EL components." '
                'style="display:inline-block;margin:0 4px;padding:0 6px;border-radius:3px;font-size:0.75rem;'
                'background:#888;color:#fff">Greek BERT stops reading here</span>'
                '<span class="after-bert" style="background:rgba(128,128,128,0.10)">'
            )
            out.append(html.escape(text[cut:b]))
            cut_done = True
        else:
            out.append(html.escape(text[a:b]))

    for s, e, cs in spans:
        if e <= start or s >= end:
            continue
        s, e = max(s, start), min(e, end)
        plain(pos, s)
        tip = "; ".join(tips.get(c, c) for c in cs)
        dim = bool(focus) and not (set(cs) & focus)
        out.append(_mark(text, s, e, cs, colour, tip, dim))
        pos = e
    plain(pos, end)
    if cut is not None and cut_done:
        out.append("</span>")
    return "".join(out)


def legend_html(
    codes: Sequence[str], spans: List[Span], colour: Dict[str, str], tips: Dict[str, str], focus: Optional[set] = None
) -> str:
    shown = {c for _, _, cs in spans for c in cs}
    chips = []
    for c in codes:
        style = f"background:{colour[c]};color:#222;padding:1px 6px;border-radius:3px;margin:0 4px 4px 0;display:inline-block;font-size:0.85rem"
        if c not in shown:
            style += ";opacity:0.45;text-decoration:line-through"
        elif focus and c not in focus:
            style += ";opacity:0.3"
        title = tips.get(c, c) + ("" if c in shown else " (no single phrase in the text)")
        chips.append(
            f'<span class="chip" data-code="{html.escape(c, quote=True)}" title="{html.escape(title, quote=True)}" '
            f'style="{style}">{html.escape(c)}</span>'
        )
    return "".join(chips)
