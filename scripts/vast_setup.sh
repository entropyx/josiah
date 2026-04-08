#!/bin/bash
# Vast.ai instance setup script for Demantiq training.
#
# Reads GITHUB_TOKEN from .env file (or pass as argument).
#
# Usage (on vast.ai after SSH):
#   # First time:
#   cd /workspace/josiah && bash scripts/vast_setup.sh
#
#   # Or from scratch (if .env has GITHUB_TOKEN):
#   # 1. scp your .env to the instance first
#   # 2. Then run this script
#
# Locally before launching:
#   git push origin feature/implement-v1

set -e

BRANCH="feature/implement-v1"
REPO="entropyx/josiah"
WORKSPACE="/workspace"

# --- Resolve token: argument > .env > environment ---
TOKEN="${1:-}"

if [ -z "$TOKEN" ]; then
    # Try .env in current dir, then in repo dir
    for envfile in ".env" "$WORKSPACE/josiah/.env"; do
        if [ -f "$envfile" ]; then
            TOKEN=$(grep -E '^GITHUB_TOKEN=' "$envfile" | cut -d= -f2- | tr -d '"' | tr -d "'")
            if [ -n "$TOKEN" ]; then
                echo "  Token loaded from $envfile"
                break
            fi
        fi
    done
fi

if [ -z "$TOKEN" ]; then
    TOKEN="${GITHUB_TOKEN:-}"
fi

echo "================================================"
echo "  Demantiq Vast.ai Setup"
echo "================================================"

# --- Clone or pull ---
if [ -d "$WORKSPACE/josiah/.git" ]; then
    echo "[1/4] Pulling latest code..."
    cd "$WORKSPACE/josiah"
    if [ -n "$TOKEN" ]; then
        git remote set-url origin "https://${TOKEN}@github.com/${REPO}.git"
    fi
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git reset --hard "origin/$BRANCH"
else
    if [ -z "$TOKEN" ]; then
        echo "ERROR: First-time setup requires a GitHub token."
        echo "Set GITHUB_TOKEN in .env or pass as argument:"
        echo "  bash scripts/vast_setup.sh YOUR_TOKEN"
        exit 1
    fi
    echo "[1/4] Cloning repo..."
    cd "$WORKSPACE"
    git clone "https://${TOKEN}@github.com/${REPO}.git" -b "$BRANCH"
    cd josiah
fi

# Copy .env from local if it was passed via scp
if [ -f "$WORKSPACE/.env" ] && [ ! -f "$WORKSPACE/josiah/.env" ]; then
    cp "$WORKSPACE/.env" "$WORKSPACE/josiah/.env"
    echo "  Copied .env into repo"
fi

# --- Install ---
echo "[2/4] Installing dependencies..."
pip install -e ".[neural]" -q 2>&1 | tail -1

# --- Verify GPU ---
echo "[3/4] Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected! Training will be slow.')
"

# --- Ready ---
echo "[4/4] Setup complete!"
echo ""
echo "================================================"
echo "  Ready to train. Run one of:"
echo ""
echo "  # PFN model (100K scenarios, ~1-2 hrs on GPU)"
echo "  python scripts/train_pfn.py --n-train 100000 --n-epochs 100 --random-eval 10 --batch-size 32 --patience 30"
echo ""
echo "  # Quick test (1K scenarios, ~5 min)"
echo "  python scripts/train_pfn.py --n-train 1000 --n-epochs 50 --random-eval 5 --batch-size 32"
echo ""
echo "  # Copy results back (run from LOCAL machine):"
echo "  # scp -P PORT -i ~/.ssh/id_ed25519_vastai -r root@HOST:/workspace/josiah/neural_output ./vast_results/"
echo ""
echo "  # DESTROY instance when done (dashboard or: python scripts/vast_train.py destroy)"
echo "================================================"
