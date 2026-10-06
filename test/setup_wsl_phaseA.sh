#!/usr/bin/env bash
# Phase A: venv + torch + clone submodules + download checkpoints
# Chay trong WSL Ubuntu-22.04. Khong can sudo.
# Khong sua gi trong apc/ ngoai viec tao apc/vision_modules/src/ (dung dung cho setup_vision_modules.sh goc).

PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
VENV_DIR="$HOME/apc_env"
SRC_DIR="$PROJECT_DIR/apc/vision_modules/src"
CKPT_DIR="$SRC_DIR/checkpoints"

step() { echo ""; echo "=============== $1 ==============="; }
ok()   { echo "STEP_OK: $1"; }
fail() { echo "STEP_FAIL: $1"; }

step "A0. Info"
echo "PROJECT_DIR=$PROJECT_DIR"
ls "$PROJECT_DIR/apc" >/dev/null 2>&1 && ok "project dir reachable" || { fail "project dir NOT reachable"; exit 1; }

step "A1. Create venv at $VENV_DIR"
if [ -f "$VENV_DIR/bin/activate" ]; then
    ok "venv already exists"
else
    python3 -m venv "$VENV_DIR" && ok "venv created" || { fail "venv creation"; exit 1; }
fi
source "$VENV_DIR/bin/activate"
python -V
pip install --upgrade pip setuptools wheel >/dev/null 2>&1 && ok "pip upgraded" || fail "pip upgrade"

step "A2. Install torch 2.4.1 + torchvision 0.19.1 (cu124)"
if python -c "import torch" 2>/dev/null; then
    ok "torch already installed: $(python -c 'import torch;print(torch.__version__)')"
else
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 \
        && ok "torch installed" || fail "torch install"
fi
python -c "import torch; print('torch', torch.__version__, '| cuda avail:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

step "A3. Clone vision module submodules (https, khong dung SSH)"
mkdir -p "$SRC_DIR"
cd "$SRC_DIR" || exit 1

clone_if_missing() {
    local url="$1"; local dir="$2"
    if [ -d "$dir" ]; then
        ok "$dir already cloned"
    else
        git clone --depth 1 "$url" "$dir" && ok "cloned $dir" || fail "clone $dir"
    fi
}

clone_if_missing https://github.com/IDEA-Research/GroundingDINO.git GroundingDINO
clone_if_missing https://github.com/apple/ml-depth-pro.git ml-depth-pro
clone_if_missing https://github.com/SpatialVision/Orient-Anything.git orient_anything
clone_if_missing https://github.com/facebookresearch/omni3d.git omni3d

step "A4. Download checkpoints (~5GB) vao $CKPT_DIR"
mkdir -p "$CKPT_DIR"

dl() {
    local url="$1"; local name="$2"
    if [ -s "$CKPT_DIR/$name" ]; then
        echo "exists: $name ($(du -h "$CKPT_DIR/$name" | cut -f1))"
        ok "$name already downloaded"
    else
        wget -c -q --show-progress -O "$CKPT_DIR/$name" "$url" && ok "downloaded $name" || fail "download $name"
    fi
}

dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth sam_vit_h_4b8939.pth
dl https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth groundingdino_swint_ogc.pth
dl https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt depth_pro.pt

step "A5. Summary"
echo "--- checkpoints ---"
ls -lh "$CKPT_DIR" 2>/dev/null
echo "--- submodules ---"
ls -d "$SRC_DIR"/*/ 2>/dev/null
echo ""
echo "PHASE_A_DONE"
