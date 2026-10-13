"""Phrase-level segmenter baseline powered by spaCy."""

from .segmenter import segment_phrases, build_nlp
from .dssp import split_text_around_token, get_random_token # [TODO]: check
from .segmenter_multilan import Segmenter # [TODO]: check

__all__ = ["segment_phrases", "build_nlp"]
__version__ = "0.1.0"
__author__ = "Marco Tumaini"
