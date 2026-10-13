"""
Public API for the spaCy-based phrase segmenter.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import spacy
from spacy.language import Language

from .rules import (
    build_phrase_matcher,
    collect_fixed_expressions,
    collect_named_entities,
    collect_noun_chunks,
    collect_prep_complements,
    collect_verb_groups,
    fill_gaps,
    is_hard_boundary,
    resolve_overlaps,
)

# Default model – change to en_core_web_trf for higher quality (slower)
MODEL_NAME = "en_core_web_md"


@lru_cache(maxsize=1)
def build_nlp(model_name: str = MODEL_NAME) -> Language:
    """
    Load the spaCy pipeline once and attach the PhraseMatcher.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' not found.\n"
            f"Install it with:\n"
            f"    python -m spacy download {model_name}"
        ) from exc

    # Attach matcher so we do not rebuild it on every call
    nlp._phrase_matcher = build_phrase_matcher(nlp)  # type: ignore[attr-defined]
    return nlp


def segment_phrases(text: str, nlp: Language | None = None) -> List[str]:
    """
    Segment *text* into ordered phrase-level units.

    Parameters
    ----------
    text :
        Input string (may contain multiple sentences).
    nlp :
        Optional pre-loaded spaCy pipeline. If omitted a cached
        pipeline is used.

    Returns
    -------
    list[str]
        Ordered list of phrases that together reconstruct the original
        text (whitespace between phrases is not preserved).
    """
    if nlp is None:
        nlp = build_nlp()

    doc = nlp(text)
    matcher = nlp._phrase_matcher  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # 1. Collect candidate multi-word spans from every heuristic
    # ------------------------------------------------------------------
    candidates = []
    candidates += collect_noun_chunks(doc)
    candidates += collect_named_entities(doc)
    candidates += collect_fixed_expressions(doc, matcher)
    candidates += collect_verb_groups(doc)
    candidates += collect_prep_complements(doc)

    # ------------------------------------------------------------------
    # 2. Discard any candidate that crosses a hard boundary
    # ------------------------------------------------------------------
    clean: list[tuple[int, int]] = []
    for start, end in candidates:
        if any(is_hard_boundary(doc[i]) for i in range(start, end)):
            continue
        # Also reject spans that begin or end on a hard boundary token
        if is_hard_boundary(doc[start]) or (end > start + 1 and is_hard_boundary(doc[end - 1])):
            continue
        clean.append((start, end))

    # ------------------------------------------------------------------
    # 3. Resolve overlaps (prefer longer spans) and fill gaps
    # ------------------------------------------------------------------
    selected = resolve_overlaps(clean, len(doc))
    final_spans = fill_gaps(selected, len(doc))

    # ------------------------------------------------------------------
    # 4. Materialise the surface strings (preserve original casing)
    # ------------------------------------------------------------------
    phrases: List[str] = []
    for start, end in final_spans:
        # Reconstruct the exact original text of the span
        span = doc[start:end]
        phrases.append(span.text)
    return phrases
