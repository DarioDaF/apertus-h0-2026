"""Phrase-level segmentation with spaCy.

This file is self-contained for Google Colab. Upload it as
``phrase_segmenter_EN.py`` and run this in a cell:

    %run phrase_segmenter_EN.py

It installs spaCy and the small English parser model if they are not already
available, then asks for text and prints one phrase unit per bracketed span.
The functions can also be imported and reused from another notebook cell.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass


MODEL_NAME = "en_core_web_sm"
HYPHENATED_WORD = re.compile(r"^[A-Za-z]+(?:-[A-Za-z]+)+$").match


def _install_if_missing() -> None:
    """Install the runtime dependencies when the file is run in Colab."""
    if importlib.util.find_spec("spacy") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "spacy"])
    if importlib.util.find_spec(MODEL_NAME) is None:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", MODEL_NAME],
        )


#_install_if_missing()

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span


NEGATIONS = {"not", "never", "no", "neither", "nor", "without"}
MODALS = {"must", "may", "shall", "can", "should", "might", "could"}

# Add domain-specific expressions here. Longer expressions should be listed
# before shorter expressions when they overlap.
FIXED_EXPRESSIONS = (
    "in accordance with",
    "in relation to",
    "as well as",
    "so far",
    "with respect to",
    "in terms of",
    "on behalf of",
    "as a result of",
    "subject to",
    "federal law",
    "code of obligations",
    "the code of obligations",
    "come into force",
    "comes into force",

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
)

@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    priority: int
    label: str

    @property
    def length(self) -> int:
        return self.end - self.start


def _is_blocked(doc: Doc, start: int, end: int, label: str = "") -> bool:
    """Return whether a candidate violates an unconditional split rule."""
    for token in doc[start:end]:
        text = token.text.casefold()
        if text in NEGATIONS or text in MODALS:
            return True
        # Explicit fixed expressions, such as "so far" and "as well as",
        # override the general rule that modifiers are separate units.
        if token.pos_ in {"ADJ", "ADV"} and label != "FIXED":
            return True
    return False


def _fixed_expression_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []
    lowered = [token.text.casefold() for token in doc]
    for expression in FIXED_EXPRESSIONS:
        words = expression.split()
        width = len(words)
        for start in range(len(doc) - width + 1):
            if lowered[start : start + width] == words:
                candidates.append(Candidate(start, start + width, 100, "FIXED"))
    return candidates


def _named_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []
    for ent in doc.ents:
        if ent.end - ent.start <= 1:
            continue
        start = ent.start
        # A determiner attached directly to a multiword name belongs to the
        # name unit: [the Federal Council].
        if start > 0 and doc[start - 1].dep_ == "det" and doc[start - 1].head.i == ent.start:
            start -= 1
        candidates.append(Candidate(start, ent.end, 95, ent.label_ or "ENTITY"))

    # Consecutive proper nouns are useful when the model did not label an
    # unfamiliar institutional or multiword name as an entity.
    start: int | None = None
    for token in doc:
        if token.pos_ == "PROPN":
            start = token.i if start is None else start
        elif start is not None:
            if token.i - start > 1:
                candidates.append(Candidate(start, token.i, 90, "PROPER_NAME"))
            start = None
    if start is not None and len(doc) - start > 1:
        candidates.append(Candidate(start, len(doc), 90, "PROPER_NAME"))
    return candidates


def _date_and_number_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []
    for ent in doc.ents:
        if ent.label_ in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL"}:
            candidates.append(Candidate(ent.start, ent.end, 92, ent.label_))
    return candidates


def _dependency_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []

    # Determiner + noun, and noun+noun compounds. These intentionally do not
    # include adjectives, which are always separate units.
    for token in doc:
        if token.dep_ == "det" and token.head.pos_ in {"NOUN", "PROPN"}:
            if token.i + 1 == token.head.i:
                candidates.append(Candidate(token.i, token.head.i + 1, 50, "DET_NOUN"))
        if token.dep_ == "compound" and token.head.pos_ in {"NOUN", "PROPN"}:
            left, right = sorted((token.i, token.head.i))
            if right == left + 1:
                candidates.append(Candidate(left, right + 1, 55, "COMPOUND"))

    # Verb + auxiliaries/passive auxiliaries. Modal and negation tokens are
    # excluded, so [must] [not] [be approved] remains properly separated.
    for verb in doc:
        if verb.pos_ not in {"VERB", "AUX"}:
            continue
        related = [
            child
            for child in verb.children
            if child.dep_ in {"aux", "auxpass"} and child.i < verb.i
            and child.text.casefold() not in MODALS
            and child.text.casefold() not in NEGATIONS
        ]
        if related:
            start = min(child.i for child in related)
            if start + 1 == verb.i or all(
                doc[index].dep_ in {"aux", "auxpass"}
                and doc[index].text.casefold() not in MODALS
                and doc[index].text.casefold() not in NEGATIONS
                for index in range(start, verb.i)
            ):
                candidates.append(Candidate(start, verb.i + 1, 60, "VERB_AUX"))

        # Infinitive "to" + verb.
        if verb.morph.get("VerbForm") == ["Inf"] and verb.i > 0:
            previous = doc[verb.i - 1]
            if previous.text.casefold() == "to":
                candidates.append(Candidate(previous.i, verb.i + 1, 65, "INFINITIVE"))

        # Phrasal particles and fixed prepositions.
        for child in verb.children:
            if child.dep_ in {"prt", "prep"} and child.i == verb.i + 1:
                candidates.append(Candidate(verb.i, child.i + 1, 65, "VERB_PARTICLE"))

    # Preposition + a bare complement. A determiner or modifier makes the
    # complement independent, e.g. [with] [the procedure].
    for token in doc:
        if token.pos_ != "ADP" or token.i + 1 >= len(doc):
            continue
        complement = doc[token.i + 1]
        if complement.pos_ in {"NOUN", "PROPN", "PRON", "NUM"}:
            if complement.dep_ in {"pobj", "pcomp", "obl"}:
                candidates.append(Candidate(token.i, complement.i + 1, 45, "PREP_COMPLEMENT"))
    return candidates


def segment_doc(doc: Doc) -> list[Span]:
    """Return non-overlapping phrase units for a parsed spaCy document."""
    candidates = (
        _fixed_expression_candidates(doc)
        + _date_and_number_candidates(doc)
        + _named_candidates(doc)
        + _dependency_candidates(doc)
    )
    candidates = [
        candidate
        for candidate in candidates
        if candidate.length > 1
        and not _is_blocked(
            doc, candidate.start, candidate.end, candidate.label
        )
    ]
    candidates.sort(key=lambda item: (-item.priority, -item.length, item.start))

    occupied = [False] * len(doc)
    selected: list[Candidate] = []
    for candidate in candidates:
        if any(occupied[candidate.start : candidate.end]):
            continue
        selected.append(candidate)
        for index in range(candidate.start, candidate.end):
            occupied[index] = True

    selected.sort(key=lambda item: item.start)
    units: list[Span] = []
    cursor = 0
    for candidate in selected:
        while cursor < candidate.start:
            units.append(doc[cursor : cursor + 1])
            cursor += 1
        units.append(doc[candidate.start : candidate.end])
        cursor = candidate.end
    while cursor < len(doc):
        units.append(doc[cursor : cursor + 1])
        cursor += 1
    return units


@Language.component("phrase_segmenter")
def phrase_segmenter(doc: Doc) -> Doc:
    doc.spans["phrase_units"] = segment_doc(doc)
    return doc


def build_pipeline(model: str = MODEL_NAME) -> Language:
    """Load a spaCy model and add the phrase segmentation component."""
    nlp = spacy.load(model)
    nlp.tokenizer = spacy.tokenizer.Tokenizer(
        nlp.vocab,
        rules=nlp.Defaults.tokenizer_exceptions,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=nlp.tokenizer.infix_finditer,
        token_match=HYPHENATED_WORD,
    )
    if "phrase_segmenter" not in nlp.pipe_names:
        nlp.add_pipe("phrase_segmenter", last=True)
    return nlp
