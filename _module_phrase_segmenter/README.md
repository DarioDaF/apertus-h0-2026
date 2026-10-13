# Phrase Segmenter (spaCy baseline)

Baseline implementation of **phrase-level segmentation** using spaCy.
It approximates the linguistic merge rules that were previously expressed
as an LLM system prompt.

## Goal

Segment text into ordered phrase-level units (never whole clauses/sentences).

```text
Input : The Federal Council has not approved the new law in accordance with the procedure.
Output: ["The Federal Council", "has not approved", "the", "new", "law", "in accordance with", "the procedure"]
```

## Supported merge heuristics (baseline) 


| Rule | How it is approximated |
|------|------------------------|
| 1. Determiner + noun | spaCy `noun_chunks` |
| 2. Proper / institutional names | NER (`ORG`, `PERSON`, `GPE` …) + noun chunks |
| 3. Noun–noun compounds | `noun_chunks` + compound dependency |
| 4. Verb groups (aux, neg, particles, “to”) | Custom dependency walk from the main verb |
| 5. Preposition + complement | Limited – only when the complement is a short noun chunk |
| 6. Fixed expressions | `PhraseMatcher` list (easy to extend) |
| 7. Dates / numbers | NER `DATE` / `CARDINAL` / `PERCENT` / `MONEY` / `QUANTITY` |

**Hard constraints kept**

- Never merge across commas, coordinating conjunctions or clause boundaries.
- Preserve original casing and spelling exactly.
- When in doubt → leave tokens separate.

This is intentionally a **baseline**. The LLM prompt was more precise on several edge cases; the spaCy version is deterministic, offline and fast.

[TODO] : To be updated since we switched to language dependant rules.

## Quick start

# Baseline
```bash
# 1. Install
pip install -r requirements.txt
python -m spacy download en_core_web_md

# 2. Run examples
python -m phrase_segmenter.cli

# 3. Segment one sentence
python -m phrase_segmenter.cli "The Federal Council has not approved the new law in accordance with the procedure."
```

# YT Apertus segmentation pipeline
python -m examples.run_examples_csv_multilan

## Project layout

```
phrase_segmenter_spacy/
├── README.md
├── requirements.txt
├── pyproject.toml
├── phrase_segmenter/
│   ├── __init__.py
│   ├── segmenter.py      # public API: segment_phrases()
│   ├── rules.py          # matchers + merge logic
│   └── cli.py            # command-line entry point
├── examples/
│   └── run_examples.py
└── tests/
    └── test_segmenter.py
```

## Extending the baseline

1. **Add more fixed expressions** → edit `FIXED_EXPRESSIONS` in `rules.py`.
2. **Improve verb-group detection** → refine `_find_verb_groups()`.
3. **Stricter preposition merging** → extend the preposition logic in `_collect_candidate_spans()`.
4. **Switch model** → change `MODEL_NAME` (e.g. `en_core_web_trf` for higher quality, slower).

## Comparison with the LLM approach

| Aspect              | spaCy baseline          | LLM prompt              |
|---------------------|-------------------------|-------------------------|
| Speed               | milliseconds            | seconds–minutes         |
| Determinism         | yes                     | no                      |
| Offline             | yes                     | no                      |
| Configuration effort| higher (code rules)     | lower (natural language)|
| Edge-case fidelity  | lower (baseline)        | higher (prompt rules)   |

Use this repo when you need a fast, reproducible starting point that you can iteratively harden with classic NLP techniques.
