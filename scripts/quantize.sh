#!/bin/bash
# =============================================================
# quantize.sh
# Convert merged HuggingFace model → GGUF Q4_K_M for deployment
# Target: 4GB VRAM machine via llama.cpp or Ollama
# =============================================================

set -e  # exit on any error

MERGED_MODEL_DIR="merged/glm-ocr-invoice-merged"
OUTPUT_F16="merged/glm-ocr-invoice-f16.gguf"
OUTPUT_Q4KM="merged/glm-ocr-invoice-q4km.gguf"
LLAMA_CPP_DIR="llama.cpp"  # path to compiled llama.cpp

echo "======================================="
echo " GLM-OCR → GGUF Quantization Pipeline"
echo "======================================="

# -- Step 1: Clone & compile llama.cpp if not present ----------
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "[1/4] Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
else
    echo "[1/4] llama.cpp already present, skipping clone"
fi

echo "[2/4] Compiling llama.cpp with CUDA support..."
cd "$LLAMA_CPP_DIR"
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --config Release -j "$(nproc)"
cd ..

# -- Step 2: Convert to FP16 GGUF -----------------------------
echo "[3/4] Converting to FP16 GGUF..."
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_MODEL_DIR" \
    --outtype f16 \
    --outfile "$OUTPUT_F16"
echo "    → $OUTPUT_F16"

# -- Step 3: Quantize to Q4_K_M -------------------------------
echo "[4/4] Quantizing to Q4_K_M (sweet spot: ~1.5GB VRAM)..."
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$OUTPUT_F16" \
    "$OUTPUT_Q4KM" \
    Q4_K_M

echo ""
echo "✅ Done!"
echo "   FP16 GGUF : $OUTPUT_F16"
echo "   Q4_K_M    : $OUTPUT_Q4KM  (use this for 4GB VRAM deployment)"
echo ""
echo "Next: copy $OUTPUT_Q4KM to your deployment machine."
echo "      Run inference with: python scripts/inference.py --mode gguf"
