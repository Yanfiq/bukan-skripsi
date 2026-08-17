import argparse
import json
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig, T5ForConditionalGeneration, T5Tokenizer


MODEL_NAME = "google/madlad400-10b-mt"
DEFAULT_INPUT = Path("dataset/text_json_final/combined.json")
DEFAULT_OUTPUT = Path("dataset/text_json_final/combined_translated.json")


def build_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )


def load_model_and_tokenizer(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set --device cpu or run on a GPU-enabled machine.")

    quantization_config = build_quantization_config() if device == "cuda" else None
    model_kwargs = {
        "device_map": "cuda" if device == "cuda" else None,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def translate_text(model, tokenizer, text: str, device: str, max_new_tokens: int = 256) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    prompt = f"<2id> {text}"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


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
