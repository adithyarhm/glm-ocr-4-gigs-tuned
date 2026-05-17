"""
prepare_dataset.py

Builds LLaMA-Factory compatible dataset (sharegpt format)
from raw invoice images + JSON annotation files.

Usage:
    python scripts/prepare_dataset.py \
        --images_dir data/raw \
        --labels_dir data/annotations \
        --output data/invoice_dataset.json
"""

import argparse
import json
import os
from pathlib import Path

# ----------------------------------------------------------------
# JSON schema used in every prompt — model must match this structure
# ----------------------------------------------------------------
INVOICE_SCHEMA = {
    "invoice_no": "",
    "date": "",
    "vendor": {
        "name": "",
        "address": "",
        "npwp": ""
    },
    "bill_to": {
        "name": "",
        "address": ""
    },
    "items": [
        {
            "description": "",
            "qty": 0,
            "unit_price": 0,
            "subtotal": 0
        }
    ],
    "subtotal": 0,
    "tax_rate": "",
    "tax_amount": 0,
    "total": 0,
    "currency": ""
}

SYSTEM_PROMPT = (
    "You are an invoice data extraction expert. "
    "Given an invoice image, extract all key information. "
    "Return ONLY a valid JSON object — no explanation, no markdown, no extra text."
)


def build_prompt() -> str:
    return (
        f"Extract all invoice information from this image and return a valid JSON object "
        f"matching exactly this schema:\n{json.dumps(INVOICE_SCHEMA, indent=2, ensure_ascii=False)}"
    )


def build_sample(image_path: str, annotation: dict) -> dict:
    """
    Build one sharegpt-format sample.
    image_path must be an ABSOLUTE or relative-to-cwd path to the image file.
    annotation is the ground-truth dict (loaded from JSON label file).
    """
    # Remove 'image' key from annotation if present (metadata, not output)
    output = {k: v for k, v in annotation.items() if k != "image"}

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": build_prompt()}
                ]
            },
            {
                "role": "assistant",
                "content": json.dumps(output, ensure_ascii=False)
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare invoice fine-tuning dataset")
    parser.add_argument("--images_dir",  default="data/raw",          help="Directory of invoice images")
    parser.add_argument("--labels_dir",  default="data/annotations",  help="Directory of JSON annotation files")
    parser.add_argument("--output",      default="data/invoice_dataset.json", help="Output dataset JSON path")
    parser.add_argument("--img_exts",    default="jpg,jpeg,png,webp",  help="Comma-separated image extensions")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    img_exts   = {f".{e.strip().lower()}" for e in args.img_exts.split(",")}

    if not images_dir.exists():
        print(f"[WARN] Images directory not found: {images_dir}")
        print("       Place invoice images in data/raw/ and re-run.")

    samples = []
    skipped = 0

    # Find all annotation JSON files
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        print(f"[ERROR] No annotation files found in {labels_dir}")
        return

    for label_file in label_files:
        with open(label_file, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        # Resolve image path from annotation or by matching stem
        img_name = annotation.get("image", None)
        if img_name:
            img_path = images_dir / img_name
        else:
            # Try to find image with same stem as label file
            found = None
            for ext in img_exts:
                candidate = images_dir / (label_file.stem + ext)
                if candidate.exists():
                    found = candidate
                    break
            img_path = found

        if img_path is None or not img_path.exists():
            print(f"[SKIP] No image found for label: {label_file.name}")
            skipped += 1
            continue

        sample = build_sample(str(img_path.resolve()), annotation)
        samples.append(sample)
        print(f"[OK] {label_file.name} → {img_path.name}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Dataset built: {len(samples)} samples → {output_path}")
    if skipped:
        print(f"⚠️  Skipped: {skipped} labels (no matching image)")


if __name__ == "__main__":
    main()
