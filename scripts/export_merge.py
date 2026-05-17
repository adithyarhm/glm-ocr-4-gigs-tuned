"""
export_merge.py

Merge LoRA adapter into base GLM-OCR weights.
Output is a standard HuggingFace model directory,
ready for conversion to GGUF.

Usage:
    python scripts/export_merge.py
    python scripts/export_merge.py \
        --adapter saves/glm-ocr-invoice-lora \
        --output  merged/glm-ocr-invoice-merged
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", default="zai-org/GLM-OCR",              help="Base model ID or path")
    parser.add_argument("--adapter",    default="saves/glm-ocr-invoice-lora",   help="Path to LoRA adapter")
    parser.add_argument("--output",     default="merged/glm-ocr-invoice-merged", help="Output directory")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"[ERROR] Adapter not found: {adapter_path}")
        print("        Run train.py first to generate the adapter.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Merging adapter from: {adapter_path}")
    print(f"[INFO] Output to:            {output_path}")

    cmd = [
        "llamafactory-cli", "export",
        "--model_name_or_path",   args.base_model,
        "--adapter_name_or_path", str(adapter_path),
        "--template",             "glm4v",
        "--finetuning_type",      "lora",
        "--export_dir",           str(output_path),
        "--export_size",          "2",
        "--export_legacy_format", "false",
        "--trust_remote_code",    "true",
    ]

    subprocess.run(cmd, check=True)
    print(f"\n✅ Merged model saved to: {output_path}")
    print("   Next step: run scripts/quantize.sh")


if __name__ == "__main__":
    main()
