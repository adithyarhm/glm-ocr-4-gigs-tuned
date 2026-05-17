# GLM-OCR Fine-Tuned for Receipt/Invoice Extraction

> Fine-tuning `zai-org/GLM-OCR` (0.9B) for receipt & invoice structured JSON extraction,  
> optimized for deployment on **4GB VRAM** hardware via GGUF Q4_K_M quantization.

## Architecture

```
Receipt Image → GLM-OCR (fine-tuned) → Structured JSON
```

No separate OCR engine needed — GLM-OCR handles vision + extraction end-to-end.

## Hardware

| Stage | Hardware | VRAM Usage |
|---|---|---|
| Fine-tuning | RTX 4080 Super (16GB) | ~6-8 GB (LoRA FP16) |
| Deployment | Any GPU ≥4GB VRAM | ~1.5 GB (GGUF Q4_K_M) |

## Dataset Format

This project uses **SROIE-format** annotations: word-level bounding quads with semantic category tags.

| Category | Description |
|---|---|
| `menu.nm` | Item name |
| `menu.cnt` | Item quantity |
| `menu.price` | Item total price |
| `sub_total.subtotal_price` | Subtotal before tax |
| `sub_total.service_price` | Service charge |
| `sub_total.tax_price` | Tax amount |
| `sub_total.etc` | Rounding / misc |
| `total.total_price` | Grand total |

## Output JSON Schema

```json
{
  "items": [
    {"description": "Nasi Campur Bali", "qty": 1, "total": 75000},
    {"description": "Bbk Bengil Nasi",  "qty": 1, "total": 125000}
  ],
  "subtotal":    1346000,
  "service":     100950,
  "tax":         144695,
  "rounding":    "Rounding -45",
  "grand_total": 1591600,
  "currency":    "IDR"
}
```

## Project Structure

```
├── data/
│   ├── raw/                         # Receipt images (.jpg/.png)
│   ├── annotations/                 # SROIE JSON annotation files
│   ├── dataset_info.json            # LLaMA-Factory dataset registry
│   ├── invoice_dataset_train.json   # Generated training set
│   └── invoice_dataset_val.json     # Generated validation set
├── configs/
│   └── finetune.yaml                # LLaMA-Factory training config
├── scripts/
│   ├── convert_sroie_to_training.py # SROIE → sharegpt format converter ⭐
│   ├── train.py                     # Fine-tuning launcher
│   ├── export_merge.py              # Merge LoRA adapter → full model
│   ├── quantize.sh                  # Convert to GGUF Q4_K_M
│   └── inference.py                 # Test inference (HF + GGUF modes)
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/adithyarhm/glm-ocr-4-gigs-tuned
cd glm-ocr-4-gigs-tuned
pip install -r requirements.txt

# Install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory && pip install -e ".[torch,metrics,bitsandbytes]" && cd ..
```

### 2. Prepare data

```bash
# Place files:
#   data/raw/receipt_00000.jpg
#   data/annotations/receipt_00000.json  (SROIE format)

# Convert SROIE annotations → LLaMA-Factory training dataset
python scripts/convert_sroie_to_training.py
# Output: data/invoice_dataset_train.json + data/invoice_dataset_val.json
```

### 3. Fine-tune

```bash
python scripts/train.py
# or directly:
llamafactory-cli train configs/finetune.yaml
```

### 4. Merge + Export + Quantize

```bash
python scripts/export_merge.py
bash scripts/quantize.sh
```

### 5. Inference

```bash
# With HuggingFace merged model
python scripts/inference.py --mode hf --image data/raw/receipt_00000.jpg

# With GGUF Q4_K_M (for 4GB VRAM deployment)
python scripts/inference.py --mode gguf --image data/raw/receipt_00000.jpg
```

## References

- [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [GLM-OCR Technical Report (arXiv:2603.10910)](https://arxiv.org/abs/2603.10910)
- [SROIE Dataset Format](https://rrc.cvc.uab.es/?ch=13)
