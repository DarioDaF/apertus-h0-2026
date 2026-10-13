# Scritp to split the text.
# [TODO]: check

from typing import TypedDict, Union
import re
import random
import pandas as pd


class MyInputWithTarget(TypedDict):
    original: str
    new_prefix: str
    new_suffix: str
    new_middle: str


def get_random_token(text: Union[str, list, None]) -> str | None:
    """
    Extract usable tokens and return a randomly chosen one.
    - If `text` is a string → tokenize it and pick a random word.
    - If `text` is a list of strings → pick a random non-empty, non-punctuation-only element.
    Returns None if nothing usable is found.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    # Case 1: already a list of segments/tokens
    if isinstance(text, list):
        candidates = [
            t.strip() for t in text
            if isinstance(t, str) and t.strip() and re.search(r"[\w]", t)
        ]
        if not candidates:
            return None
        return random.choice(candidates)

    # Case 2: plain string
    if not isinstance(text, str) or not text.strip():
        return None

    words = re.findall(r"\b[\w'-]+\b", text)
    if not words:
        return None
    return random.choice(words)


def split_text_around_token(text: Union[str, list, None], token: str | None) -> MyInputWithTarget:
    """
    Split `text` around the first occurrence of `token`.

    - If `text` is a list of strings, the list is split into
      prefix (elements before the token), middle (the token itself),
      and suffix (elements after the token).
    - If `text` is a plain string, classic string split is performed.
    - If token is None or not found → whole content goes into prefix,
      middle and suffix stay empty.
    """
    # ------------------------------------------------------------------
    # Normalize input to a clean list of strings
    # ------------------------------------------------------------------
    if text is None or (isinstance(text, float) and pd.isna(text)):
        texts: list[str] = []
    elif isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        texts = [
            "" if (t is None or (isinstance(t, float) and pd.isna(t))) else str(t)
            for t in text
        ]
    else:
        texts = [str(text)]

    # ------------------------------------------------------------------
    # Helper for a single string
    # ------------------------------------------------------------------
    def _split_string(s: str) -> tuple[str, str, str]:
        if token is None:
            return s, "", ""
        idx = s.find(token)
        if idx == -1:
            return s, "", ""
        return s[:idx], token, s[idx + len(token):]

    # ------------------------------------------------------------------
    # Empty input
    # ------------------------------------------------------------------
    if not texts:
        return MyInputWithTarget(
            original=[],
            new_prefix=[],
            new_middle=[],
            new_suffix=[],
        )

    # ------------------------------------------------------------------
    # List-of-strings path (main use-case with segment_A_bonus style data)
    # ------------------------------------------------------------------
    if isinstance(text, list) or len(texts) > 1:
        if token is None:
            return MyInputWithTarget(
                original=texts,
                new_prefix=texts,
                new_middle=[],
                new_suffix=[],
            )

        try:
            idx = texts.index(token)          # first exact match
        except ValueError:
            # token not present
            return MyInputWithTarget(
                original=texts,
                new_prefix=texts,
                new_middle=[],
                new_suffix=[],
            )

        return MyInputWithTarget(
            original=texts,
            new_prefix=texts[:idx],
            new_middle=[token],
            new_suffix=texts[idx + 1:],
        )

    # ------------------------------------------------------------------
    # Single-string path (backward compatible)
    # ------------------------------------------------------------------
    prefix, middle, suffix = _split_string(texts[0])
    return MyInputWithTarget(
        original=texts[0],
        new_prefix=prefix,
        new_middle=middle,
        new_suffix=suffix,
    )