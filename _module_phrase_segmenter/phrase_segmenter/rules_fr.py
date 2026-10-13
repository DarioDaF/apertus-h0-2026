from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass


MODEL_NAME = "fr_core_news_sm"


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
    "ne", "n'", "pas", "jamais", "plus", "rien", "aucun", "aucune",
    "personne", "ni", "sans", "non",
}
MODAL_LEMMAS = {"devoir", "pouvoir", "vouloir", "falloir"}
MODAL_FORMS = {
    "dois", "doit", "devons", "devez", "doivent",
    "peux", "peut", "pouvons", "pouvez", "peuvent",
    "veux", "veut", "voulons", "voulez", "veulent",
    "faut", "fallait", "faudra", "faudrait",
}
CONJUNCTIONS = {"et", "ou", "si", "sauf", "mais", "seulement", "uniquement"}
MONTHS = {
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
}

# Longer expressions should be listed before shorter expressions when they
# overlap. Add domain-specific French expressions here as needed.
FIXED_EXPRESSIONS = (
    "conformément à",
    "en lien avec",
    "ainsi que",
    "jusqu'ici",
    "à moins que",
    "en outre",
    "en ligne",
    "a tenu compte",
    "assurance maladie",
    "impôt sur le revenu",
    "surveillance du marché",
    "Conseil fédéral",
    "Département fédéral des finances",
    "Union européenne",
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
    text = token.text.casefold()
    return text in NEGATIONS or text.startswith("n'")


def _merge_elided_clitics(doc: Doc) -> None:
    """Keep elided clitics with their following word, e.g. ``n'a``."""
    clitics = {"c'", "d'", "j'", "l'", "m'", "n'", "s'", "t'"}
    with doc.retokenize() as retokenizer:
        for index in range(len(doc) - 1):
            if (
                doc[index].text.casefold() in clitics
                and doc[index + 1].is_alpha
            ):
                retokenizer.merge(doc[index : index + 2])


def _is_blocked(doc: Doc, start: int, end: int, label: str = "") -> bool:
    """Return whether a candidate violates an unconditional split rule."""
    for token in doc[start:end]:
        if _is_negation(token) or _is_modal(token):
            return True
        if token.text.casefold() in CONJUNCTIONS:
            return True
        if (
            token.pos_ in {"ADJ", "ADV"}
            and label not in {"FIXED", "ENTITY", "ORG", "PROPER_NAME", "NUMBER"}
        ):
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
        if token.pos_ != "NUM":
            continue
        end = token.i + 1
        if end < len(doc) and doc[end].text == "%":
            candidates.append(Candidate(token.i, end + 1, 92, "NUMBER"))
            continue
        if (
            end + 1 < len(doc)
            and doc[end].text.casefold() in MONTHS
            and doc[end + 1].pos_ == "NUM"
        ):
            end += 2
        elif (
            end + 3 < len(doc)
            and doc[end].pos_ == "NOUN"
            and doc[end + 1].pos_ == "ADP"
            and doc[end + 2].pos_ in {"NOUN", "PROPN"}
        ):
            end += 3
        else:
            continue
        start = token.i
        while start > 0 and doc[start - 1].pos_ in {"ADP", "DET"}:
            start -= 1
        candidates.append(Candidate(start, end, 92, "NUMBER"))
    return candidates


def _dependency_candidates(doc: Doc) -> list[Candidate]:
    candidates: list[Candidate] = []

    # Article/demonstrative/possessive + noun, only when directly adjacent.
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

    # Verb groups: auxiliaries, passive auxiliaries, clitics and reflexives.
    for verb in doc:
        if verb.pos_ not in {"VERB", "AUX"} or _is_modal(verb):
            continue
        start = verb.i
        while start > 0:
            previous = doc[start - 1]
            if _is_modal(previous) or _is_negation(previous):
                break
            if (
                previous.dep_.startswith("aux")
                or previous.dep_ in {"expl", "obj", "iobj"}
            ):
                start -= 1
                continue
            break
        if start < verb.i:
            candidates.append(Candidate(start, verb.i + 1, 60, "VERB_GROUP"))

        # ``de`` or ``à`` before an infinitive.
        if verb.morph.get("VerbForm") == ["Inf"] and verb.i > 0:
            previous = doc[verb.i - 1]
            if previous.text.casefold() in {"de", "à"}:
                candidates.append(Candidate(previous.i, verb.i + 1, 65, "INFINITIVE"))

        # Verb + fixed preposition, e.g. ``collaborer avec``.
        for child in verb.children:
            if child.dep_ in {"prt", "prep"} and child.i == verb.i + 1:
                candidates.append(Candidate(verb.i, child.i + 1, 65, "VERB_PREPOSITION"))

    # Preposition + complement. Determiners are allowed, but an adjective or
    # adverb makes the complement an independent unit.
    for token in doc:
        if token.pos_ not in {"ADP", "DET"} or token.i + 1 >= len(doc):
            continue
        end = token.i + 1
        while end < len(doc) and doc[end].pos_ in {"DET", "PRON"}:
            end += 1
        if end < len(doc) and doc[end].pos_ in {"NOUN", "PROPN", "NUM"}:
            complement = doc[end]
            if (
                complement.dep_ == "ROOT"
                or complement.dep_.startswith("obl")
                or complement.dep_.startswith("nmod")
                or complement.dep_ in {"pobj", "pcomp"}
            ):
                candidates.append(Candidate(token.i, end + 1, 45, "PREP_COMPLEMENT"))
    return candidates


def segment_doc(doc: Doc) -> list[Span]:
    """Return non-overlapping phrase units for a parsed spaCy document."""
    _merge_elided_clitics(doc)
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


@Language.component("phrase_segmenter_fr")
def phrase_segmenter(doc: Doc) -> Doc:
    doc.spans["phrase_units"] = segment_doc(doc)
    return doc


def build_pipeline(model: str = MODEL_NAME) -> Language:
    """Load a spaCy model and add the phrase segmentation component."""
    nlp = spacy.load(model)
    if "phrase_segmenter_fr" not in nlp.pipe_names:
        nlp.add_pipe("phrase_segmenter_fr", last=True)
    return nlp
