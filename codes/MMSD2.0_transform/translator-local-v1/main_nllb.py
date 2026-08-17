import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch

MODEL_NAME = "facebook/nllb-200-3.3B"
DEFAULT_INPUT = Path("dataset/text_json_final/combined.json")
DEFAULT_OUTPUT = Path("dataset/text_json_id/dataset_translated_nllb-200.json")

def load_model_and_tokenizer(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set --device cpu or run on a GPU-enabled machine.")

    model_kwargs = {
        "device_map": "cuda" if device == "cuda" else None,
        "dtype": torch.float16 if device == "cuda" else torch.float32,
    }

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = "eng_Latn"
    return model, tokenizer


def translate_text(model, tokenizer, text: str, device: str, max_new_tokens: int = 256) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    inputs = tokenizer(text, return_tensors="pt").to(device)

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("ind_Latn")
    )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


def load_records(input_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {input_path}, found {type(data).__name__}")

    return data


def format_json_object(obj: dict) -> str:
    formatted = json.dumps(obj, ensure_ascii=False, indent=2)
    return "  " + formatted.replace("\n", "\n  ")


def translate_dataset(input_path: Path, output_path: Path, device: str, limit: int | None = None, in_place: bool = False):
    records = load_records(input_path)

    if limit is not None:
        records = records[:limit]

    final_output = input_path if in_place else output_path

    model, tokenizer = load_model_and_tokenizer(device)
    model.eval()

    final_output.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    with final_output.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        for idx, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Record at index {idx - 1} is not an object: {record!r}")

            original_text = record.get("text")
            new_record = dict(record)

            if isinstance(original_text, str):
                translated_text = translate_text(model, tokenizer, original_text, device)
                new_record["text_translated"] = translated_text
            else:
                new_record["text_translated"] = ""

            if not first:
                f.write(",\n")

            f.write(format_json_object(new_record))
            first = False
            total_written += 1
            f.flush()

        f.write("\n]\n")

    print(f"Saved {total_written} translated records to {final_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Translate MMSD2.0 JSON text data with Madlad400 10B MT.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the source JSON file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to the output JSON file.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device to run translation on.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to translate for a quick test run.")
    parser.add_argument("--in-place", action="store_true", help="Write back to the same input file instead of the default output path.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    translate_dataset(
        input_path=input_path,
        output_path=output_path,
        device=args.device,
        limit=args.limit,
        in_place=args.in_place,
    )


if __name__ == "__main__":
    main()
