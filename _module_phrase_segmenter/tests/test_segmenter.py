"""Minimal smoke tests for the spaCy phrase segmenter."""

from __future__ import annotations
from typing import Callable
from phrase_segmenter import segment_phrases

test_list = []
def test_fn(f: Callable[[], None]):
    global test_list
    test_list.append(f)


@test_fn
def test_basic_sentence():
    text = "The Federal Council has not approved the new law."
    phrases = segment_phrases(text)
    assert isinstance(phrases, list)
    assert all(isinstance(p, str) for p in phrases)
    # Reconstruct roughly the same content
    joined = " ".join(phrases)
    assert "Federal Council" in joined or "Federal" in joined
    assert "has not approved" in joined or "approved" in joined


@test_fn
def test_fixed_expression():
    text = "Decision taken in accordance with the procedure."
    phrases = segment_phrases(text)
    # The fixed expression should appear as one unit (or at least be present)
    assert any("accordance" in p for p in phrases)


@test_fn
def test_empty_string():
    assert segment_phrases("") == []


@test_fn
def test_preserves_casing():
    text = "OFCOM will be required to cooperate."
    phrases = segment_phrases(text)
    assert any("OFCOM" in p for p in phrases)


@test_fn
def test_segmenter_multilan():
    from phrase_segmenter import Segmenter

    for lang, sentence in [
        ('EN', 'I wonder what the EU central bank thinks about this...'),
        ('IT', 'Chissa cosa ne pensa la banca centrale Europea...'),
        ('DE', 'Wer weiß, was die Europäische Zentralbank wohl dazu denkt...'),
        ('FR', 'Qui sait ce qu\'en pense la Banque centrale européenne...'),
    ]:
        s = Segmenter(lang)
        r = s.segment(sentence)
        print(f'[{lang}] -> {r}')
        assert isinstance(r, list)
        assert len(r) > 1


if __name__ == '__main__':
    passed_tot = 0
    for test in test_list:
        print(f'TEST: {test.__name__}')
        passed = False
        try:
            test()
            passed = True
        except Exception as e:
            print(f'\033[31m  FAILED\033[0m | {e}')
        if passed:
            passed_tot += 1
            print(f'\033[32m  PASSED\033[0m')
