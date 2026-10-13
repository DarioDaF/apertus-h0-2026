#!/usr/bin/env python3
"""Run the same example sentences that were used in the original LLM prompt."""

from phrase_segmenter import segment_phrases

EXAMPLES = [
    (
        "The Federal Council has not approved the new law in accordance with the procedure.",
        # Expected-ish (LLM version for reference)
        # ["The Federal Council", "has not approved", "the", "new", "law", "in accordance with", "the procedure"]
    ),
    (
        "The Council took the experience made so far into account.",
        # ["The Council", "took", "the experience", "made", "so far", "into account"]
    ),
    (
        "Also, online marketplaces or platforms will be required to cooperate with OFCOM in relation to market surveillance.",
        # ["Also", "online", "marketplaces", "or", "platforms", "will be required", "to cooperate with", "OFCOM", "in relation to", "market surveillance"]
    ),
]


def main() -> None:
    print("=== spaCy baseline – comparison examples ===\n")
    for text, *_ in EXAMPLES:
        phrases = segment_phrases(text)
        print(f"IN : {text}")
        print(f"OUT: {phrases}")
        print()


if __name__ == "__main__":
    main()
