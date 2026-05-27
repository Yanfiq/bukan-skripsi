from __future__ import annotations

import argparse
import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "datasets"
    / "MMSD2.0"
    / "data"
    / "dataset_image"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "datasets"
    / "MMSD2.0"
    / "data"
    / "dataset_image_cleaned"
)


@dataclass(frozen=True)
class OCRMetrics:
    detections: int
    text_characters: int
    text_area_ratio: float
    average_confidence: float


def iter_images(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def polygon_area(points: list[list[float]] | list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def should_remove_text_heavy_image(
    image_path: Path,
    reader: Any,
    min_text_area_ratio: float,
    min_detections: int,
    min_confidence: float,
    min_text_characters: int,
) -> tuple[bool, OCRMetrics]:
    detections = reader.readtext(str(image_path), detail=1, paragraph=False)

    if not detections:
        return False, OCRMetrics(0, 0, 0.0, 0.0)

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a transitive dependency of EasyOCR.
        raise RuntimeError("Pillow is required to inspect image dimensions.") from exc

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    image_area = float(image_width * image_height) if image_width and image_height else 0.0
    if image_area == 0.0:
        return False, OCRMetrics(0, 0, 0.0, 0.0)

    text_area = 0.0
    total_confidence = 0.0
    text_characters = 0
    valid_detections = 0

    for bbox, text, confidence in detections:
        cleaned_text = text.strip()
        if cleaned_text:
            text_characters += len(cleaned_text)

        if confidence >= min_confidence and cleaned_text:
            valid_detections += 1

        text_area += polygon_area(bbox)
        total_confidence += float(confidence)

    text_area_ratio = text_area / image_area
    average_confidence = total_confidence / len(detections)
    metrics = OCRMetrics(
        detections=len(detections),
        text_characters=text_characters,
        text_area_ratio=text_area_ratio,
        average_confidence=average_confidence,
    )

    remove_image = (
        valid_detections >= min_detections and text_area_ratio >= min_text_area_ratio
    ) or (
        text_characters >= min_text_characters and average_confidence >= min_confidence
    )

    return remove_image, metrics


def build_reader(languages: list[str], gpu: bool) -> Any:
    easyocr = importlib.import_module("easyocr")
    return easyocr.Reader(languages, gpu=gpu)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove text-heavy images from MMSD2.0 dataset_image using EasyOCR."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the original dataset images.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "datasets" / "MMSD2.0" / "data" / "dataset_image_filenames.txt",
        help="Text file where kept image filenames will be written (one per line).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting.",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Optional directory for images flagged as text-heavy.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="EasyOCR languages to load, for example: --languages en id",
    )
    parser.add_argument(
        "--min-text-area-ratio",
        type=float,
        default=0.12,
        help="Minimum OCR bounding-box area ratio for removing an image.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=2,
        help="Minimum number of confident OCR detections for removal.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum OCR confidence for a detection to count toward the filter.",
    )
    parser.add_argument(
        "--min-text-characters",
        type=int,
        default=20,
        help="Fallback text-length threshold for text-heavy images.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force EasyOCR to run on CPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matches without writing any files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    append_mode = bool(args.append)
    quarantine_dir = args.quarantine_dir.expanduser().resolve() if args.quarantine_dir else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not args.cpu:
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            use_gpu = False
        else:
            use_gpu = bool(torch.cuda.is_available())
    else:
        use_gpu = False

    reader = build_reader(args.languages, gpu=use_gpu)

    kept_count = 0
    removed_count = 0
    skipped_count = 0

    if not args.dry_run:
        if output_file.parent:
            output_file.parent.mkdir(parents=True, exist_ok=True)
        if quarantine_dir is not None:
            quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output file handle. Use append or overwrite depending on flag.
    out_fh = None
    if not args.dry_run:
        mode = "a" if append_mode else "w"
        out_fh = output_file.open(mode, encoding="utf-8")

    for image_path in iter_images(input_dir):
        try:
            remove_image, metrics = should_remove_text_heavy_image(
                image_path=image_path,
                reader=reader,
                min_text_area_ratio=args.min_text_area_ratio,
                min_detections=args.min_detections,
                min_confidence=args.min_confidence,
                min_text_characters=args.min_text_characters,
            )
        except Exception as exc:
            skipped_count += 1
            print(f"[skip] {image_path.name}: {exc}")
            continue

        status = "remove" if remove_image else "keep"
        print(
            f"[{status}] {image_path.name} | detections={metrics.detections} | "
            f"chars={metrics.text_characters} | area_ratio={metrics.text_area_ratio:.3f} | "
            f"avg_conf={metrics.average_confidence:.3f}"
        )

        if remove_image:
            removed_count += 1
            if not args.dry_run and quarantine_dir is not None:
                shutil.copy2(image_path, quarantine_dir / image_path.name)
        else:
            kept_count += 1
            if not args.dry_run and out_fh is not None:
                out_fh.write(image_path.name + "\n")

    if out_fh is not None:
        out_fh.close()

    print(
        f"Finished. wrote={kept_count}, removed={removed_count}, skipped={skipped_count}, "
        f"dry_run={args.dry_run}, gpu={use_gpu}, output_file={output_file}"
    )


if __name__ == "__main__":
    main()
