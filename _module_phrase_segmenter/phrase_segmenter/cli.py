"""
Command-line entry point.

Usage
-----
python -m phrase_segmenter.cli "Your sentence here."
python -m phrase_segmenter.cli                  # runs built-in examples
"""

from __future__ import annotations

import argparse
import sys

from .segmenter import segment_phrases


EXAMPLES = [
    "The Federal Council has not approved the new law in accordance with the procedure.",
    "The Council took the experience made so far into account.",
    "Also, online marketplaces or platforms will be required to cooperate with OFCOM in relation to market surveillance.",
    "1 Every person has legal capacity.\n\n"
    "2 Accordingly, within the limits of the law, every person has the same "
    "capacity to have rights and obligations.",
    "Where the law confers discretion on the court or makes reference to an "
    "assessment of the circumstances or to good cause, the court must reach "
    "its decision in accordance with the principles of justice and equity.",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phrase-level segmenter (spaCy baseline)"
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to segment (if omitted, runs built-in examples)",
    )
    args = parser.parse_args(argv)

    if args.text is not None:
        phrases = segment_phrases(args.text)
        print(phrases)
        return

    print("=== Phrase segmenter (spaCy baseline) ===\n")
    for ex in EXAMPLES:
        phrases = segment_phrases(ex)
        print(f"IN : {ex}")
        print(f"OUT: {phrases}\n")


if __name__ == "__main__":
    main()
