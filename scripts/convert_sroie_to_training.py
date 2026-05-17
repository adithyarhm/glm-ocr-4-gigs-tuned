"""
convert_sroie_to_training.py

Converts SROIE-format receipt annotations (word-level bounding quads
with category tags) into a LLaMA-Factory sharegpt training dataset.

Expected input structure per sample:
  data/raw/receipt_00001.jpg          <- receipt image
  data/annotations/receipt_00001.json <- SROIE annotation

The annotation JSON must have this structure:
  {
    "valid_line": [
      {
        "category": "menu.nm" | "menu.cnt" | "menu.price" |
                    "sub_total.subtotal_price" | "sub_total.service_price" |
                    "sub_total.tax_price" | "sub_total.etc" |
                    "total.total_price",
        "group_id": <int>,
        "words": [{"text": "...", "quad": {...}}, ...]
      },
      ...
    ]
  }

Usage:
    python scripts/convert_sroie_to_training.py
    python scripts/convert_sroie_to_training.py \
        --images_dir  data/raw \
        --labels_dir  data/annotations \
        --output      data/invoice_dataset.json \
        --val_split   0.1
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------
# Target JSON schema — model must output exactly this structure
# ---------------------------------------------------------------
INVOICE_SCHEMA = {
    "items": [
        {"description": "", "qty": 0, "total": 0}
    ],
    "subtotal":    0,
    "service":     0,
    "tax":         0,
    "rounding":    "",
    "grand_total": 0,
    "currency":    ""
}

SYSTEM_PROMPT = (
    "You are a receipt data extraction expert. "
    "Given a receipt image, extract all transaction information. "
    "Return ONLY a valid JSON object — no explanation, no markdown, no extra text."
)

USER_PROMPT = (
    f"Extract all receipt information from this image and return a valid JSON object "
    f"matching exactly this schema:\n"
    f"{json.dumps(INVOICE_SCHEMA, indent=2, ensure_ascii=False)}"
)


# ---------------------------------------------------------------
# SROIE annotation parser
# ---------------------------------------------------------------
def parse_price(text: str) -> int:
    """Convert '1,346,000' or '75,000.' to int."""
    cleaned = re.sub(r'[^\d]', '', text)
    return int(cleaned) if cleaned else 0


def parse_count(cnt_tokens: list) -> int:
    """Extract numeric qty from token list like ['x', '1'] or ['3', 'x']."""
    for token in cnt_tokens:
        if re.match(r'^\d+$', token.strip()):
            return int(token.strip())
    return 1


def sroie_to_structured_json(annotation: dict) -> dict:
    """Parse SROIE annotation dict → clean structured JSON (ground truth)."""
    groups = defaultdict(lambda: {"cnt": [], "nm": [], "price": []})
    totals = {}

    for line in annotation.get("valid_line", []):
        gid  = line.get("group_id")
        cat  = line.get("category", "")
        text = " ".join(w["text"] for w in line.get("words", []))

        if cat == "menu.cnt":
            groups[gid]["cnt"].extend(text.split())
        elif cat == "menu.nm":
            groups[gid]["nm"].append(text)
        elif cat == "menu.price":
            groups[gid]["price"].append(text)
        elif cat == "sub_total.subtotal_price":
            # text like "Sub-Total 1,346,000" — take last token
            totals["subtotal"] = parse_price(text.split()[-1])
        elif cat == "sub_total.service_price":
            totals["service"] = parse_price(text.split()[-1])
        elif cat == "sub_total.tax_price":
            totals["tax"] = parse_price(text.split()[-1])
        elif cat == "sub_total.etc":
            totals["rounding"] = text.strip()
        elif cat == "total.total_price":
            totals["grand_total"] = parse_price(text.split()[-1])

    items = []
    for gid, fields in sorted(groups.items()):
        qty   = parse_count(fields["cnt"])
        name  = " ".join(fields["nm"]).strip()
        price = parse_price(fields["price"][0]) if fields["price"] else 0
        if name:  # skip empty groups
            items.append({"description": name, "qty": qty, "total": price})

    return {
        "items":       items,
        "subtotal":    totals.get("subtotal",    0),
        "service":     totals.get("service",     0),
        "tax":         totals.get("tax",          0),
        "rounding":    totals.get("rounding",    ""),
        "grand_total": totals.get("grand_total", 0),
        "currency":    "IDR"
    }


# ---------------------------------------------------------------
# Build one sharegpt training sample
# ---------------------------------------------------------------
def build_training_sample(image_path: str, structured_json: dict) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": USER_PROMPT}
                ]
            },
            {
                "role": "assistant",
                "content": json.dumps(structured_json, ensure_ascii=False)
            }
        ]
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert SROIE annotations to LLaMA-Factory dataset")
    parser.add_argument("--images_dir",  default="data/raw",               help="Directory of receipt images")
    parser.add_argument("--labels_dir",  default="data/annotations",       help="Directory of SROIE JSON annotation files")
    parser.add_argument("--output",      default="data/invoice_dataset.json", help="Output training dataset path")
    parser.add_argument("--val_split",   type=float, default=0.1,          help="Fraction for validation split (0 = no split)")
    parser.add_argument("--img_exts",    default="jpg,jpeg,png,webp",      help="Comma-separated image extensions to search")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    img_exts   = {f".{e.strip().lower()}" for e in args.img_exts.split(",")}

    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        print(f"[ERROR] No annotation .json files found in {labels_dir}")
        sys.exit(1)

    samples, skipped = [], 0

    for label_file in label_files:
        with open(label_file, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        # Resolve matching image
        img_path = None
        for ext in img_exts:
            candidate = images_dir / (label_file.stem + ext)
            # Also handle: receipt_00000.json -> receipt_00000-2.jpg pattern
            candidate2 = images_dir / (label_file.stem + "-2" + ext)
            if candidate.exists():
                img_path = candidate
                break
            elif candidate2.exists():
                img_path = candidate2
                break

        if img_path is None:
            print(f"[SKIP] No image found for: {label_file.name}")
            skipped += 1
            continue

        structured = sroie_to_structured_json(annotation)

        # Sanity check: skip samples with no items
        if not structured["items"]:
            print(f"[SKIP] No menu items parsed from: {label_file.name}")
            skipped += 1
            continue

        sample = build_training_sample(str(img_path.resolve()), structured)
        samples.append(sample)
        print(f"[OK] {label_file.name} | items={len(structured['items'])} | total={structured['grand_total']:,}")

    if not samples:
        print("[ERROR] No valid samples built. Check your data directories.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.val_split > 0 and len(samples) >= 10:
        import random
        random.seed(42)
        random.shuffle(samples)
        n_val   = max(1, int(len(samples) * args.val_split))
        n_train = len(samples) - n_val
        train_samples = samples[:n_train]
        val_samples   = samples[n_train:]

        train_path = output_path.parent / (output_path.stem + "_train.json")
        val_path   = output_path.parent / (output_path.stem + "_val.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Dataset split complete:")
        print(f"   Train : {len(train_samples)} samples → {train_path}")
        print(f"   Val   : {len(val_samples)} samples → {val_path}")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Dataset built: {len(samples)} samples → {output_path}")

    if skipped:
        print(f"⚠️  Skipped: {skipped} annotation files (no matching image or empty items)")


if __name__ == "__main__":
    main()
