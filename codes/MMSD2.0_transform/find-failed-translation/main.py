import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


DEFAULT_INPUT = Path("dataset/text_json_id/dataset_translated_madlad.json")
DEFAULT_OUTPUT = Path("dataset/text_json_id/dataset_failed_translation_madlad.json")
DEFAULT_THRESHOLD = 0.9


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def similarity(first_text, second_text):
    return SequenceMatcher(
        None,
        normalize_text(first_text),
        normalize_text(second_text),
    ).ratio()


def main():
    parser = argparse.ArgumentParser(
        description="Find translations that are too similar to the source text."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Similarity threshold from 0 to 1 (default: 0.9).",
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    with args.input.open("r", encoding="utf-8") as dataset_file:
        records = json.load(dataset_file)

    failed_records = []
    for record in records:
        score = similarity(record.get("text", ""), record.get("text_translated", ""))
        if score >= args.threshold:
            failed_records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(failed_records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(f"Similarity threshold: {args.threshold:.2f}")
    print(f"Failed translations: {len(failed_records)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
