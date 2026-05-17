# GLM-OCR Fine-Tuned for Invoice Extraction

> Fine-tuning `zai-org/GLM-OCR` (0.9B) for invoice/receipt structured JSON extraction, optimized for deployment on 4GB VRAM hardware.

## Architecture

```
Invoice Image → GLM-OCR (fine-tuned) → Structured JSON
```

No separate OCR engine needed — GLM-OCR handles vision + extraction end-to-end.

## Hardware

| Stage | Hardware | VRAM Usage |
|---|---|---|
| Fine-tuning | RTX 4080 Super (16GB) | ~6-8 GB (LoRA FP16) |
| Deployment | Any GPU with 4GB VRAM | ~1.5 GB (Q4_K_M GGUF) |

## Project Structure

```
├── data/
│   ├── raw/                  # Raw invoice images
│   ├── annotations/          # JSON ground truth labels
│   ├── dataset_info.json     # LLaMA-Factory dataset registry
│   └── invoice_dataset.json  # Formatted training dataset
├── configs/
│   └── finetune.yaml         # LLaMA-Factory training config
├── scripts/
│   ├── prepare_dataset.py    # Build dataset from raw images + labels
│   ├── train.py              # Launch fine-tuning
│   ├── export_merge.py       # Merge LoRA adapter → full weights
│   ├── quantize.sh           # Convert to GGUF Q4_K_M
│   └── inference.py          # Test inference (HF + GGUF)
├── saves/                    # Training checkpoints (gitignored)
├── merged/                   # Merged model weights (gitignored)
├── requirements.txt
└── .gitignore
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
```bash
python scripts/prepare_dataset.py \
  --images_dir data/raw \
  --labels_dir data/annotations \
  --output data/invoice_dataset.json
```

### 3. Fine-tune
```bash
python scripts/train.py
# or directly:
chamafactory-cli train configs/finetune.yaml
```

### 4. Merge + Export
```bash
python scripts/export_merge.py
```

### 5. Quantize for deployment
```bash
bash scripts/quantize.sh
```

### 6. Run inference
```bash
# HuggingFace format (after fine-tuning, before quantize)
python scripts/inference.py --mode hf --image path/to/invoice.jpg

# GGUF format (after quantize, for 4GB VRAM deployment)
python scripts/inference.py --mode gguf --image path/to/invoice.jpg
```

## Output JSON Schema

```json
{
  "invoice_no": "INV-2024-001",
  "date": "15/05/2024",
  "vendor": {
    "name": "PT Maju Jaya",
    "address": "Jl. Sudirman No. 1, Jakarta",
    "npwp": "01.234.567.8-901.000"
  },
  "bill_to": {
    "name": "PT Pembeli Setia",
    "address": "Jl. Gatot Subroto No. 5, Bandung"
  },
  "items": [
    {
      "description": "Laptop ASUS ROG",
      "qty": 2,
      "unit_price": 8500000,
      "subtotal": 17000000
    }
  ],
  "subtotal": 17000000,
  "tax_rate": "11%",
  "tax_amount": 1870000,
  "total": 18870000,
  "currency": "IDR"
}
```

## References
- [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [GLM-OCR Technical Report](https://arxiv.org/abs/2603.10910)
