from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass


MODEL_NAME = "it_core_news_sm"


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


NEGATIONS = {
    "non", "né", "ne", "mai", "senza", "nessun", "nessuna", "nessuno",
    "nessune", "nessuni", "alcun", "alcuna", "alcuno", "alcune", "alcuni",
    "niente", "nulla",
}
MODAL_LEMMAS = {"dovere", "potere", "volere"}
MODAL_FORMS = {
    "devo", "devi", "deve", "dobbiamo", "dovete", "devono",
    "posso", "puoi", "può", "possiamo", "potete", "possono",
    "voglio", "vuoi", "vuole", "vogliamo", "volete", "vogliono",
}

# Add domain-specific expressions here. Longer expressions should be listed
# before shorter expressions when they overlap.
FIXED_EXPRESSIONS = (
    "in conformità con",
    "in relazione",
    "a meno che",
    "ha tenuto conto",
    "consiglio federale",
    "dipartimento federale delle finanze",
    "unione europea",
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


def _is_modal(token) -> bool:
    return (
        token.text.casefold() in MODAL_FORMS
        or token.lemma_.casefold() in MODAL_LEMMAS
    )


def _is_blocked(doc: Doc, start: int, end: int, label: str = "") -> bool:
    """Return whether a candidate violates an unconditional split rule."""
    for token in doc[start:end]:
        text = token.text.casefold()
        if text in NEGATIONS or _is_modal(token):
            return True
        # Explicit fixed expressions, such as "a meno che",
        # override the general rule that modifiers are separate units.
        if (
            token.pos_ in {"ADJ", "ADV"}
            and label not in {"FIXED", "ENTITY", "PROPER_NAME"}
        ):
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
                candidate_start = start
                if (
                    start > 0
                    and doc[start - 1].pos_ in {"DET", "PRON"}
                    and doc[start - 1].head.i == start
                ):
                    candidate_start -= 1
                candidates.append(
                    Candidate(candidate_start, start + width, 100, "FIXED")
                )
    return candidates


def _named_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []
    for ent in doc.ents:
        if ent.end - ent.start <= 1:
            continue
        start = ent.start
        # An article or demonstrative directly attached to a multiword name
        # belongs to the name unit: [il Consiglio federale].
        if (
            start > 0
            and doc[start - 1].pos_ in {"DET", "PRON"}
            and doc[start - 1].head.i == ent.start
        ):
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
            start = ent.start
            while start > 0 and doc[start - 1].pos_ in {"ADP", "DET", "NUM"}:
                start -= 1
            candidates.append(Candidate(start, ent.end, 92, ent.label_))
    return candidates


def _dependency_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []

    # Article/demonstrative/possessive + noun. Adjectives are deliberately not
    # included: [la nuova legge], for example, must be split.
    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        start = token.i
        while start > 0 and doc[start - 1].pos_ in {"DET", "PRON"}:
            start -= 1
        if start < token.i and all(
            doc[index].pos_ in {"DET", "PRON"}
            for index in range(start, token.i)
        ):
            candidates.append(Candidate(start, token.i + 1, 50, "NOMINAL"))

        # Noun+noun compounds.
        if token.dep_ == "compound" and token.head.pos_ in {"NOUN", "PROPN"}:
            left, right = sorted((token.i, token.head.i))
            if right == left + 1:
                candidates.append(Candidate(left, right + 1, 55, "COMPOUND"))

    # Verb groups: auxiliaries, passive auxiliaries, clitics, and reflexive or
    # impersonal "si". Modal verbs and negations stay outside the group.
    for verb in doc:
        if verb.pos_ not in {"VERB", "AUX"}:
            continue
        start = verb.i
        while start > 0:
            previous = doc[start - 1]
            if _is_modal(previous) or previous.text.casefold() in NEGATIONS:
                break
            if previous.dep_ in {"aux", "auxpass", "expl", "obj", "iobj"}:
                start -= 1
                continue
            if previous.text.casefold() == "si":
                start -= 1
                continue
            break
        if start < verb.i:
            candidates.append(Candidate(start, verb.i + 1, 60, "VERB_GROUP"))

        # "di", "a", or "da" + infinitive.
        if verb.morph.get("VerbForm") == ["Inf"] and verb.i > 0:
            previous = doc[verb.i - 1]
            if previous.text.casefold() in {"di", "a", "da"}:
                # "è tenuto a" is a modal expression; the infinitive remains
                # separate: [è tenuto a] [presentare].
                if (
                    previous.text.casefold() == "a"
                    and verb.i >= 3
                    and doc[verb.i - 2].text.casefold()
                    in {"tenuto", "tenuta", "tenuti", "tenute"}
                    and doc[verb.i - 3].text.casefold() == "è"
                ):
                    candidates.append(
                        Candidate(verb.i - 3, verb.i, 66, "MODAL_GROUP")
                    )
                candidates.append(Candidate(previous.i, verb.i + 1, 65, "INFINITIVE"))

        # Verb + a fixed preposition, e.g. [collaborare con].
        for child in verb.children:
            if child.dep_ in {"prt", "prep"} and child.i == verb.i + 1:
                candidates.append(Candidate(verb.i, child.i + 1, 65, "VERB_PREPOSITION"))

    # Preposition + a bare complement. A modifier makes the complement
    # independent; fused forms such as "della" are still one token.
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


@Language.component("phrase_segmenter_it")
def phrase_segmenter(doc: Doc) -> Doc:
    doc.spans["phrase_units"] = segment_doc(doc)
    return doc


def build_pipeline(model: str = MODEL_NAME) -> Language:
    """Load a spaCy model and add the phrase segmentation component."""
    nlp = spacy.load(model)
    if "phrase_segmenter_it" not in nlp.pipe_names:
        nlp.add_pipe("phrase_segmenter_it", last=True)
    return nlp
