#!/bin/bash
# RunPod setup script for LLM Meditation project (scratchpad eval)
# Run this on a fresh RunPod A100 80GB instance.
#
# Prerequisites:
#   - Upload 3 data files via Jupyter FIRST (see below)
#   - HuggingFace token set (for Gemma 3 gated model)
#
# Usage:
#   bash /workspace/model-meditations/scripts/setup_runpod.sh

set -e

echo "=== LLM Meditation: RunPod Setup ==="
echo ""

# ── 1. Environment ──────────────────────────────────────────────────────────
export HF_HOME=/workspace/.cache/huggingface
export PROJECT=/workspace/model-meditations
mkdir -p $HF_HOME

echo "[1/5] Environment configured"
echo "  HF_HOME=$HF_HOME"
echo "  PROJECT=$PROJECT"

# ── 2. Clone repo (or pull latest) ─────────────────────────────────────────
if [ -d "$PROJECT/.git" ]; then
    echo "[2/5] Repo exists, pulling latest..."
    cd $PROJECT && git pull
else
    echo "[2/5] Cloning repo..."
    cd /workspace
    git clone https://github.com/mkenney2/model-meditations.git model-meditations
fi
cd $PROJECT

# ── 3. Install dependencies ────────────────────────────────────────────────
echo "[3/5] Installing dependencies..."
pip install -e ".[dev]" 2>&1 | tail -5

# ── 4. Check data files ────────────────────────────────────────────────────
echo "[4/5] Checking data files..."

MISSING=0

if [ ! -f "data/axis_vectors/gemma_3_27b_it_axis.pt" ]; then
    echo "  MISSING: data/axis_vectors/gemma_3_27b_it_axis.pt"
    MISSING=1
fi

if [ ! -f "data/calibration/normal_range.pt" ]; then
    echo "  MISSING: data/calibration/normal_range.pt"
    MISSING=1
fi

if [ ! -f "data/feature_cache/gemma-3-27b-it_31-gemmascope-2-res-65k.json" ]; then
    echo "  MISSING: data/feature_cache/gemma-3-27b-it_31-gemmascope-2-res-65k.json"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "  *** Upload missing files via Jupyter before running the eval! ***"
    echo "  All 3 files total ~1.3 MB."
    echo ""
    echo "  mkdir -p data/axis_vectors data/calibration data/feature_cache"
    echo "  Then upload via Jupyter file browser."
else
    echo "  All data files present."
fi

# ── 5. Download model (if not cached) ──────────────────────────────────────
echo "[5/5] Downloading Gemma 3 27B-IT (if not cached)..."
python -c "
from huggingface_hub import snapshot_download
import os
model_dir = os.path.join(os.environ['HF_HOME'], 'hub', 'models--google--gemma-3-27b-it')
if os.path.exists(model_dir):
    print('  Model already cached.')
else:
    print('  Downloading google/gemma-3-27b-it (this takes ~15 min)...')
    snapshot_download('google/gemma-3-27b-it')
    print('  Download complete.')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the scratchpad drift eval (quick test, 1 script per domain):"
echo "  cd $PROJECT"
echo "  python scripts/run_eval.py --condition scratchpad --eval drift -n 1"
echo ""
echo "To run the full scratchpad drift eval (5 scripts per domain):"
echo "  cd $PROJECT"
echo "  python scripts/run_eval.py --condition scratchpad --eval drift -n 5"
echo ""
echo "To analyze results after eval completes:"
echo "  python scripts/analyze.py"
