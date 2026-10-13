#!/usr/bin/env python3
"""
Read examples.csv, run segment_phrases() on the 'text_A' column,
store the result (or an error string) in a new column 'segment_A_bonus',
and write the result to examples_segmented.csv.
"""

import csv
from pathlib import Path

from phrase_segmenter import segment_phrases
from phrase_segmenter import split_text_around_token, get_random_token

from phrase_segmenter.segmenter_multilan import Segmenter

# Same folder as this script
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "examples.csv"
OUTPUT_CSV = SCRIPT_DIR / "examples_segmented.csv"

# apply segmenter function
def process_row_segmenter(text: str):
    """Run segment_phrases and return a Python list, or an error string on failure."""
    try:
        return segment_phrases(text)
    except Exception as exc:
        return f"ERROR: {exc}"

# apply splitting function
def process_row_split(text: str):
    """Run split_text_around_token and return a MyInputWithTarget object, or an error string on failure."""
    try:
        token = get_random_token(text)
        return split_text_around_token(text, token)
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> None:

    # import segmenter by language
    segmenter_en = Segmenter("EN")
    segmenter_it = Segmenter("IT")
    segmenter_de = Segmenter("DE")
    segmenter_fr = Segmenter("FR")
    
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    with INPUT_CSV.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        if "text_A" not in reader.fieldnames:
            raise ValueError("Column 'text_A' not found in the CSV header")
        if "text_B" not in reader.fieldnames:
            raise ValueError("Column 'text_B' not found in the CSV header")            

        fieldnames = list(reader.fieldnames) + ["segment_A_bonus"] + ["segment_B_bonus"] + ["segment_A_split"]
        rows = []
        for row in reader:
            # apply segmenter function
            text_a = row.get("text_A", "")
            # text_a is always in eng.
            text_a_seg = segmenter_en(text_a)
            row["segment_A_bonus"] = text_a_seg
            text_b = row.get("text_B", "")
            # [TODO] : this need to call the right Segmenter object.
            lang_id_b = row.get("lang_B_id", "")
            if lang_id_b == "IT":
                row["segment_B_bonus"] = segmenter_it(text_b)
            elif lang_id_b == "DE":
                row["segment_B_bonus"] = segmenter_de(text_b)
            elif lang_id_b == "FR":
                row["segment_B_bonus"] = segmenter_fr(text_b)                
            else:
                row["segment_B_bonus"] = None                            
            
            # apply split funtion
            row["segment_A_split"] = process_row_split(text_a_seg)
            
            rows.append(row)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} rows.")
    print(f"Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
