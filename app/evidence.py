"""Locate text evidence for predicted codes: dictionary term hits mapped back to the original text.

The dictionary matcher works on ``normalize_text`` output (lowercased, accent-stripped,
punctuation removed, whitespace collapsed), so its offsets don't line up with the text the
user sees. ``normalize_with_offsets`` rebuilds that exact string while remembering, for every
character, where it came from.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set, Tuple

import ahocorasick

from dictionary.normalize import normalize_text, strip_accents

_KEEP = re.compile(r"[a-zA-Zα-ωΑ-Ω0-9]")
_TO_SPACE = {"–", "—", "-", "/", "\\", "\n", "\t"}


def normalize_with_offsets(text: str) -> Tuple[str, List[int]] | None:
    """Same string as ``normalize_text(text)`` plus the source index of every character.

    Returns None if the two ever disagree, so callers never highlight the wrong span.
    """
    chars: List[str] = []
    src: List[int] = []
    # Lowercase the whole text, not char by char: Greek final sigma (Σ -> ς) depends on context.
    lowered = text.lower()
    if len(lowered) != len(text):
        lowered = None
    for i, ch in enumerate(text):
        if ch == "΄":
            continue
        for c in strip_accents(lowered[i] if lowered is not None else ch.lower()):
            if c in _TO_SPACE or not _KEEP.match(c):
                c = " "
            if c == " " and (not chars or chars[-1] == " "):
                continue
            chars.append(c)
            src.append(i)
    if chars and chars[-1] == " ":
        chars.pop()
        src.pop()
    norm = "".join(chars)
    return (norm, src) if norm == normalize_text(text) else None


def build_evidence_automaton(term_code_map: Dict[str, Set[str]]) -> ahocorasick.Automaton:
    automaton = ahocorasick.Automaton()
    for term, codes in term_code_map.items():
        if term.strip():
            automaton.add_word(f" {term} ", (len(term), frozenset(codes)))
    automaton.make_automaton()
    return automaton


def find_evidence(
    text: str, automaton: ahocorasick.Automaton, codes: Iterable[str]
) -> List[Tuple[int, int, str]]:
    """``(start, end, code)`` spans in ``text`` where a dictionary term for a wanted code occurs."""
    wanted = set(codes)
    mapped = normalize_with_offsets(text)
    if not wanted or mapped is None:
        return []
    norm, src = mapped
    padded = f" {norm} "
    hits = []
    for end, (length, term_codes) in automaton.iter(padded):
        # ``end`` points at the trailing pad space; the term itself sits just before it.
        last = end - 1  # index in ``padded`` of the term's last char
        first = last - length + 1
        s, e = src[first - 1], src[last - 1] + 1  # padded -> norm -> original text
        for code in sorted(term_codes & wanted):
            hits.append((s, e, code))
    return hits
