from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass


MODEL_NAME = "de_core_news_sm"


def _install_if_missing() -> None:
    """Install runtime dependencies when the file is run in Colab."""
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
    "nicht", "kein", "keine", "keinen", "keinem", "keiner", "keines",
    "nie", "niemals", "nichts", "niemand", "weder", "noch", "ohne",
}
MODAL_LEMMAS = {"müssen", "dürfen", "können", "sollen", "wollen", "mögen"}
MODAL_FORMS = {
    "muss", "musst", "müssen", "müsst", "musste", "mussten",
    "darf", "darfst", "dürfen", "dürft", "durfte", "durften",
    "kann", "kannst", "können", "könnt", "konnte", "konnten",
    "soll", "sollst", "sollen", "sollt", "sollte", "sollten",
    "will", "willst", "wollen", "wollt", "wollte", "wollten",
    "mag", "möchte", "mögen", "mochte", "mochten",
}
CONJUNCTIONS = {"und", "oder", "wenn", "außer", "aber", "nur", "lediglich"}
MONTHS = {
    "januar", "februar", "märz", "april", "mai", "juni",
    "juli", "august", "september", "oktober", "november", "dezember",
}
YEAR_WITH_PERIOD = re.compile(r"^\d{4}\.$")
ADPOSITIONS = {
    "an", "auf", "aus", "bei", "durch", "für", "gegen", "gemäß", "in",
    "mit", "nach", "ohne", "seit", "über", "um", "unter", "von", "vor",
    "während", "wegen", "zu", "zwischen", "bis", "gemäß", "gemäss", "an",
    "im", "am", "zum", "zur",
    "vom", "beim", "ins", "ans", "aufs",
}
MODAL_PHRASES = {"ist verpflichtet", "sind verpflichtet", "war verpflichtet"}

# Longer expressions should be listed before shorter expressions when they
# overlap. Add domain-specific German expressions here as needed.
FIXED_EXPRESSIONS = (
    "in Übereinstimmung mit",
    "im Zusammenhang mit",
    "im Hinblick auf",
    "es sei denn",
    "Rechnung getragen",
    "in Kraft getreten",
    "Kauf auf Probe",
    "Europäische Union",
    "Eidgenössisches Finanzdepartement",
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


def _is_negation(token) -> bool:
    return token.text.casefold() in NEGATIONS


def _is_blocked(doc: Doc, start: int, end: int, label: str = "") -> bool:
    """Return whether a candidate violates an unconditional split rule."""
    for token in doc[start:end]:
        if _is_negation(token) or _is_modal(token):
            return True
        if token.text.casefold() in CONJUNCTIONS:
            return True
        if token.pos_ in {"ADJ", "ADV"} and label not in {
            "FIXED", "ENTITY", "PROPER_NAME", "NUMBER", "MODAL_GROUP",
        }:
            return True
        if token.is_punct:
            return True
    return False


def _fixed_expression_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []
    lowered = [token.text.casefold() for token in doc]
    for expression in FIXED_EXPRESSIONS:
        words = expression.casefold().split()
        width = len(words)
        for start in range(len(doc) - width + 1):
            if lowered[start : start + width] == words:
                candidates.append(Candidate(start, start + width, 100, "FIXED"))
    lowered = [token.text.casefold() for token in doc]
    for expression in MODAL_PHRASES:
        words = expression.casefold().split()
        width = len(words)
        for start in range(len(doc) - width + 1):
            if lowered[start : start + width] == words:
                candidates.append(Candidate(start, start + width, 101, "MODAL_GROUP"))
    return candidates


def _named_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []
    for ent in doc.ents:
        if ent.end - ent.start <= 1:
            continue
        start = ent.start
        if (
            start > 0
            and doc[start - 1].pos_ in {"DET", "PRON"}
            and doc[start - 1].head.i == ent.start
        ):
            start -= 1
        candidates.append(Candidate(start, ent.end, 95, "ENTITY"))

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
        if ent.label_ not in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL"}:
            continue
        start = ent.start
        while start > 0 and doc[start - 1].pos_ in {"ADP", "DET", "NUM"}:
            start -= 1
        candidates.append(Candidate(start, ent.end, 92, "NUMBER"))

    for token in doc:
        if token.pos_ not in {"NUM", "NOUN", "ADJ"}:
            continue
        if (
            token.pos_ == "NOUN"
            and token.text.casefold() not in MONTHS
        ):
            continue
        if token.text[:-1].isdigit() and token.text.endswith("."):
            month = token.i + 1
            if month < len(doc) and doc[month].text.casefold() in MONTHS:
                end = month + 1
                if end < len(doc) and (
                    doc[end].pos_ == "NUM"
                    or doc[end].text.isdigit()
                ):
                    end += 1
                start = token.i
                while start > 0 and doc[start - 1].pos_ in {"ADP", "DET"}:
                    start -= 1
                candidates.append(Candidate(start, end, 92, "NUMBER"))
                continue

        end = token.i + 1
        if end < len(doc) and doc[end].text.casefold() in {
            "prozent", "millionen", "million", "milliarden", "franken",
            "jahre", "jahr", "absatz",
        }:
            end += 1
            if end < len(doc) and doc[end].pos_ in {"NOUN", "PROPN", "NUM"}:
                end += 1
            candidates.append(Candidate(token.i, end, 92, "NUMBER"))
    return candidates


def _split_sentence_periods(doc: Doc) -> None:
    """Keep a sentence-final year period outside the numerical unit."""
    with doc.retokenize() as retokenizer:
        for token in list(doc):
            if (
                YEAR_WITH_PERIOD.fullmatch(token.text)
                and token.i > 0
                and doc[token.i - 1].text.casefold() in MONTHS
            ):
                retokenizer.split(
                    token,
                    [token.text[:-1], "."],
                    heads=[token.head, token.head],
                )


def _dependency_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []

    # Article/demonstrative/possessive + noun, only if directly adjacent.
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

        # Adjacent noun compounds are useful when the parser marks them.
        if token.dep_ == "compound" and token.head.pos_ in {"NOUN", "PROPN"}:
            left, right = sorted((token.i, token.head.i))
            if right == left + 1:
                candidates.append(Candidate(left, right + 1, 55, "COMPOUND"))

    # German verb groups can be ordered lexical verb + auxiliary or auxiliary
    # + lexical verb. Separable prefixes are represented as adjacent particles.
    for verb in doc:
        if verb.pos_ not in {"VERB", "AUX"} or _is_modal(verb):
            continue
        related = [
            child for child in verb.children
            if child.dep_.startswith("aux")
            and child.text.casefold() not in MODAL_FORMS
        ]
        if related:
            start = min(child.i for child in related)
            end = max(child.i for child in related + [verb]) + 1
            if end - start == len(related) + 1:
                candidates.append(Candidate(start, end, 60, "VERB_GROUP"))

        adjacent = []
        index = verb.i - 1
        while index >= 0 and doc[index].pos_ in {"VERB", "AUX"}:
            if _is_modal(doc[index]) or _is_negation(doc[index]):
                break
            adjacent.append(index)
            index -= 1
        index = verb.i + 1
        while index < len(doc) and doc[index].pos_ in {"VERB", "AUX"}:
            if _is_modal(doc[index]) or _is_negation(doc[index]):
                break
            adjacent.append(index)
            index += 1
        if adjacent:
            start = min([verb.i, *adjacent])
            end = max([verb.i, *adjacent]) + 1
            if end - start > 1:
                candidates.append(Candidate(start, end, 60, "VERB_GROUP"))

        if (
            verb.morph.get("VerbForm") == ["Inf"]
            and verb.i > 0
            and doc[verb.i - 1].text.casefold() == "zu"
        ):
            candidates.append(
                Candidate(verb.i - 1, verb.i + 1, 65, "INFINITIVE")
            )

        for child in verb.children:
            if child.dep_ in {"svp", "prt"} and child.i == verb.i + 1:
                candidates.append(Candidate(verb.i, child.i + 1, 65, "SEPARABLE_VERB"))
            if child.dep_ == "expl" and child.text.casefold() == "sich":
                left, right = sorted((child.i, verb.i))
                if right == left + 1:
                    candidates.append(Candidate(left, right + 1, 60, "REFLEXIVE"))

    # Preposition + complement. Genitive attributes are intentionally not
    # candidates because ``die Grundsätze`` and ``der Gerechtigkeit`` split.
    for token in doc:
        if token.text.casefold() not in ADPOSITIONS or token.i + 1 >= len(doc):
            continue
        end = token.i + 1
        while end < len(doc) and doc[end].pos_ in {"DET", "PRON"}:
            end += 1
        if end < len(doc) and doc[end].pos_ in {"NOUN", "PROPN", "NUM"}:
            complement = doc[end]
            candidate_end = end + 1
            if complement.pos_ == "NUM":
                number_end = end + 1
                if (
                    number_end < len(doc)
                    and doc[number_end].text.casefold()
                    in {"prozent", "millionen", "million", "milliarden", "franken"}
                ):
                    candidate_end = number_end + 1
            if (
                complement.dep_.startswith("obl")
                or complement.dep_.startswith("nmod")
                or complement.dep_ in {"ROOT", "pobj", "pcomp", "da", "oa"}
                or complement.pos_ == "NUM"
            ):
                priority = 93 if complement.pos_ == "NUM" else 56
                candidates.append(
                    Candidate(token.i, candidate_end, priority, "PREP_COMPLEMENT")
                )
    return candidates


def segment_doc(doc: Doc) -> list[Span]:
    """Return non-overlapping phrase units for a parsed spaCy document."""
    _split_sentence_periods(doc)
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
        and not _is_blocked(doc, candidate.start, candidate.end, candidate.label)
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


@Language.component("phrase_segmenter_de")
def phrase_segmenter(doc: Doc) -> Doc:
    doc.spans["phrase_units"] = segment_doc(doc)
    return doc


def build_pipeline(model: str = MODEL_NAME) -> Language:
    """Load a spaCy model and add the phrase segmentation component."""
    nlp = spacy.load(model)
    if "phrase_segmenter_de" not in nlp.pipe_names:
        nlp.add_pipe("phrase_segmenter_de", last=True)
    return nlp
