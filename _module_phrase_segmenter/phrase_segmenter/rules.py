"""
Merge rules and candidate-span collectors.

These heuristics approximate the original LLM system-prompt rules.
They are deliberately conservative: when in doubt we leave tokens separate.
"""

from __future__ import annotations

from typing import Iterable, List, Set, Tuple

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token

# ---------------------------------------------------------------------------
# Fixed multi-word expressions (Rule 6)
# ---------------------------------------------------------------------------

FIXED_EXPRESSIONS: List[str] = [
    # original prompt examples
    "in accordance with",
    "in relation to",
    "as well as",
    "so far",
    "highly qualified",
    # useful legal / administrative additions
    "subject to",
    "pursuant to",
    "with regard to",
    "with respect to",
    "in respect of",
    "for the purpose of",
    "in the event of",
    "in the case of",
    "on the basis of",
    "by virtue of",
    "in accordance",
    "in addition to",
    "as opposed to",
    "in order to",
    "in order for",
    "with a view to",
    "for the sake of",
    "in lieu of",
    "by means of",
    "in terms of",
    "with effect from",
    "as from",
    "as of",
    "up to and including",
    "not less than",
    "not more than",
    "at least",
    "at most",
]


def build_phrase_matcher(nlp: spacy.language.Language) -> PhraseMatcher:
    """Create a PhraseMatcher for the fixed expressions (case-insensitive)."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(expr) for expr in FIXED_EXPRESSIONS]
    matcher.add("FIXED", patterns)
    return matcher


# ---------------------------------------------------------------------------
# Helper: tokens that must never be merged across
# ---------------------------------------------------------------------------

def is_hard_boundary(token: Token) -> bool:
    """
    True if this token must break any potential multi-word phrase.
    Corresponds to: "Never merge across a clause boundary, conjunction or comma."
    """
    if token.text in {",", ";", ":", "—", "–", "-"}:
        return True
    if token.pos_ == "CCONJ":          # and, or, but …
        return True
    if token.dep_ in {"cc", "punct"} and token.text in {",", ";"}:
        return True
    # Sentence / clause boundaries
    if token.is_sent_start:
        return True
    return False


# ---------------------------------------------------------------------------
# Candidate span collectors
# ---------------------------------------------------------------------------

SpanTuple = Tuple[int, int]   # (start_token_idx, end_token_idx)  – end exclusive


def _span_from_tokens(tokens: List[Token]) -> SpanTuple | None:
    if not tokens:
        return None
    start = tokens[0].i
    end = tokens[-1].i + 1
    return (start, end)


def collect_noun_chunks(doc: Doc) -> List[SpanTuple]:
    """Rule 1 + 3: determiner+noun and noun compounds via spaCy noun_chunks."""
    spans = []
    for chunk in doc.noun_chunks:
        # Keep the whole chunk; later overlap resolution will prefer longer spans
        spans.append((chunk.start, chunk.end))
    return spans


def collect_named_entities(doc: Doc) -> List[SpanTuple]:
    """Rule 2 + 7: proper names, institutions, dates, numbers …"""
    interesting = {
        "ORG", "PERSON", "GPE", "LOC", "FAC",
        "DATE", "TIME", "CARDINAL", "PERCENT", "MONEY", "QUANTITY", "ORDINAL",
    }
    spans = []
    for ent in doc.ents:
        if ent.label_ in interesting:
            spans.append((ent.start, ent.end))
    return spans


def collect_fixed_expressions(doc: Doc, matcher: PhraseMatcher) -> List[SpanTuple]:
    """Rule 6: fixed multi-word expressions."""
    spans = []
    for _, start, end in matcher(doc):
        spans.append((start, end))
    return spans


def _collect_aux_chain(verb: Token) -> List[Token]:
    """
    Walk left from a verb and collect auxiliaries, negation, infinitival 'to'
    and particles that belong to the same verb group (Rule 4).
    """
    group: List[Token] = [verb]

    # left-side auxiliaries / neg / 'to'
    for tok in reversed(list(verb.lefts)):
        if tok.dep_ in {"aux", "auxpass", "neg"} or (
            tok.dep_ == "aux" and tok.lemma_ == "to"
        ) or (tok.lemma_ == "to" and tok.dep_ == "aux"):
            group.insert(0, tok)
        elif tok.dep_ == "advmod" and tok.lemma_ in {"not", "n't", "never"}:
            group.insert(0, tok)
        else:
            # stop at the first non-aux element
            break

    # particles (phrasal verbs) – usually to the right
    for tok in verb.rights:
        if tok.dep_ == "prt":
            group.append(tok)
        else:
            break

    return group


def collect_verb_groups(doc: Doc) -> List[SpanTuple]:
    """
    Rule 4: verb + auxiliaries / modals / passive / negation / particles / 'to'.
    Conservative: only merge tokens that form a contiguous span.
    """
    spans = []
    seen: Set[int] = set()

    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        if token.i in seen:
            continue

        group = _collect_aux_chain(token)
        # Only keep contiguous groups (no gaps)
        indices = [t.i for t in group]
        if indices == list(range(min(indices), max(indices) + 1)):
            span = _span_from_tokens(group)
            if span:
                spans.append(span)
                seen.update(indices)

    return spans


def collect_prep_complements(doc: Doc) -> List[SpanTuple]:
    """
    Rule 5 (limited): preposition + short complement when the complement
    has no independent modifier (very conservative heuristic).
    """
    spans = []
    for token in doc:
        if token.pos_ != "ADP":
            continue
        # Look for a short noun-chunk that is the immediate right neighbour
        if token.i + 1 >= len(doc):
            continue
        # Simple case: "to minors", "for the procedure"
        # We only merge if the following tokens form a short noun chunk
        # that starts right after the preposition.
        for chunk in doc.noun_chunks:
            if chunk.start == token.i + 1 and len(chunk) <= 4:
                # reject if the chunk contains a hard boundary
                if any(is_hard_boundary(t) for t in chunk):
                    continue
                spans.append((token.i, chunk.end))
                break
    return spans


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

def resolve_overlaps(spans: Iterable[SpanTuple], doc_len: int) -> List[SpanTuple]:
    """
    Prefer longer spans. Discard any span that crosses a hard boundary.
    Returns non-overlapping spans sorted by start index.
    """
    # Sort by length (desc) then by start (asc)
    candidates = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))

    taken = [False] * doc_len
    selected: List[SpanTuple] = []

    for start, end in candidates:
        if start >= end or end > doc_len:
            continue
        # Skip if any token already taken
        if any(taken[i] for i in range(start, end)):
            continue
        # Accept
        for i in range(start, end):
            taken[i] = True
        selected.append((start, end))

    # Sort by position
    selected.sort(key=lambda s: s[0])
    return selected


def fill_gaps(selected: List[SpanTuple], doc_len: int) -> List[SpanTuple]:
    """Insert single-token spans for every token not covered by a multi-word phrase."""
    covered = set()
    for start, end in selected:
        covered.update(range(start, end))

    result: List[SpanTuple] = []
    i = 0
    sel_idx = 0
    while i < doc_len:
        if sel_idx < len(selected) and selected[sel_idx][0] == i:
            result.append(selected[sel_idx])
            i = selected[sel_idx][1]
            sel_idx += 1
        else:
            result.append((i, i + 1))
            i += 1
    return result
