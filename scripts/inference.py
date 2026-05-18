"""
inference.py

Test inference for fine-tuned GLM-OCR.
Supports two modes:
  - hf:   HuggingFace format (merged model, requires ~2GB VRAM FP16)
  - gguf: GGUF format via llama-cpp-python (requires ~1.5GB VRAM, ideal for 4GB machines)

Usage:
    python scripts/inference.py --mode hf   --image path/to/invoice.jpg
    python scripts/inference.py --mode gguf --image path/to/invoice.jpg
    python scripts/inference.py --mode hf   --image path/to/invoice.jpg --adapter saves/glm-ocr-invoice-lora
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------
# JSON Schema (must match training schema in prepare_dataset.py)
# ---------------------------------------------------------------
# Must match the schema used in convert_sroie_to_training.py
RECEIPT_SCHEMA = {
    "items": [{"description": "", "qty": 0, "total": 0}],
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


def build_user_prompt() -> str:
    return (
        f"Extract all receipt information from this image and return a valid JSON object "
        f"matching exactly this schema:\n{json.dumps(RECEIPT_SCHEMA, indent=2, ensure_ascii=False)}"
    )


def extract_json(text: str) -> dict:
    """Robustly extract first JSON object from model output."""
    # Strip <think>...</think> reasoning blocks if present
    text = re.sub(r"</?think>", "", text).strip()
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\n?", "", text).strip()
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON parse error: {e}")
            return {"raw_output": text}
    return {"error": "No JSON found in model output", "raw_output": text}


# ---------------------------------------------------------------
# HuggingFace inference
# ---------------------------------------------------------------
def infer_hf(image_path: str, model_dir: str, adapter_path: str = None):
    import torch
    from transformers import GlmOcrForConditionalGeneration, AutoProcessor
    from PIL import Image

    print(f"[HF] Loading model from: {model_dir}")
    # AutoProcessor handles both the tokenizer and image processor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    if adapter_path:
        # Load base + LoRA adapter (for testing mid-training)
        from peft import PeftModel
        base = GlmOcrForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
    else:
        # Load merged model directly
        model = GlmOcrForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    model.eval()
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{build_user_prompt()}"}
            ]
        }
    ]

    # Step 1: render chat template to a text string (no tokenization)
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Step 2: tokenize + process image together
    inputs = processor(
        text=prompt_text,
        images=[image],
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True
    )
    return extract_json(response)


# ---------------------------------------------------------------
# GGUF inference via llama-cpp-python
# ---------------------------------------------------------------
def infer_gguf(image_path: str, model_path: str):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[ERROR] llama-cpp-python not installed.")
        print("        pip install llama-cpp-python[cuda]")
        sys.exit(1)

    print(f"[GGUF] Loading model from: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,   # offload all layers to GPU
        n_ctx=2048,
        verbose=False,
    )

    # Encode image to base64
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = "image/png" if ext == "png" else "image/jpeg"

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text",      "text": build_user_prompt()}
                ]
            }
        ],
        max_tokens=512,
        temperature=0.1,
    )

    text = response["choices"][0]["message"]["content"]
    return extract_json(text)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GLM-OCR Invoice Inference")
    parser.add_argument("--mode",    choices=["hf", "gguf"], required=True,
                        help="Inference mode: hf (HuggingFace) or gguf (llama.cpp)")
    parser.add_argument("--image",   required=True,
                        help="Path to invoice image")
    parser.add_argument("--model",   default=None,
                        help="Model path (auto-detected if omitted)")
    parser.add_argument("--adapter", default=None,
                        help="[hf mode] LoRA adapter path (optional, for mid-training test)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    if args.mode == "hf":
        model_dir = args.model or "merged/glm-ocr-invoice-merged"
        result = infer_hf(str(image_path), model_dir, args.adapter)
    else:
        model_path = args.model or "merged/glm-ocr-invoice-q4km.gguf"
        result = infer_gguf(str(image_path), model_path)

    print("\n" + "=" * 50)
    print("Extracted Invoice Data:")
    print("=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
