"""
train.py

Launch GLM-OCR fine-tuning via LLaMA-Factory.
This script is a thin wrapper — it validates the environment
then calls llamafactory-cli with configs/finetune.yaml.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/finetune.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_environment():
    """Validate that LLaMA-Factory is installed."""
    result = subprocess.run(
        ["llamafactory-cli", "--help"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[ERROR] LLaMA-Factory not found!")
        print("Install it with:")
        print("  git clone https://github.com/hiyouga/LLaMA-Factory")
        print("  cd LLaMA-Factory && pip install -e '[torch,metrics,bitsandbytes]'")
        sys.exit(1)
    print("[OK] LLaMA-Factory found")


def check_dataset():
    """Warn if train/val split files are missing."""
    import json
    splits = {
        "train": Path("data/invoice_dataset_train.json"),
        "val":   Path("data/invoice_dataset_val.json"),
    }
    for name, p in splits.items():
        if not p.exists():
            print(f"[WARN] {name} split not found at {p}")
            print("       Run convert_sroie_to_training.py first:")
            print("       python scripts/convert_sroie_to_training.py")
            sys.exit(1)
        if not p.is_file():
            print(f"[ERROR] {p} exists but is not a file (is it a directory?)")
            sys.exit(1)
        with open(p) as f:
            data = json.load(f)
        print(f"[OK] {name:5s} split: {len(data):4d} samples → {p}")


def main():
    parser = argparse.ArgumentParser(description="Launch GLM-OCR fine-tuning")
    parser.add_argument("--config", default="configs/finetune.yaml", help="Path to training config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    check_environment()
    check_dataset()

    print(f"\n🚀 Starting fine-tuning with config: {config_path}")
    print("   Monitor training loss in: saves/glm-ocr-invoice-lora/logs/\n")

    subprocess.run(
        ["llamafactory-cli", "train", str(config_path)],
        check=True
    )


if __name__ == "__main__":
    main()
