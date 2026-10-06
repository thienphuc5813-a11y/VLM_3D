#!/usr/bin/env bash
# Phase B: cai python package + build extension trong venv ~/apc_env. Chay SAU Phase A.
# Khong can sudo. Chay: bash test/setup_wsl_phaseB.sh  (tu goc repo, trong WSL)
#
# Cac quyet dinh quan trong (da kiem tra tren may nay ngay 2026-10-04):
#
# 1. MAX_JOBS=2: WSL chi co 7GB RAM / 16 core. Build pytorch3d + detectron2 voi 16 job
#    song song se OOM. (Khop voi canh bao o LOCAL_RUN_PLAN.md muc 1.7.)
#
# 2. GroundingDINO CAN nvcc, khong the build CPU-only:
#    ms_deform_attn.py:330 chi dung nhanh pure-PyTorch khi tensor o CPU. Neu tensor o GPU
#    ma _C chua build thi _C khong ton tai -> NameError luc chay. Lay nvcc bang pip wheel
#    (nvidia-cuda-nvcc-cu12) roi dung CUDA_HOME shim, de khong phai cai CUDA toolkit qua sudo.
#
# 3. pytorch3d + detectron2 build CPU-only (FORCE_CUDA=0, CUDA_HOME rong): ta chi dung phan
#    hinh hoc (look_at_view_transform, Meshes, cubercnn util) nen khong can CUDA ops,
#    va build CUDA cho hai package nay rat lau.
#
# 4. GroundingDINO build tu BAN COPY o ~/build (filesystem Linux, duong dan ASCII) thay vi
#    /mnt/c/.../"kì 4"/...: duong dan co dau cach + ky tu non-ASCII de lam hong build C++/CUDA,
#    va /mnt/c cham. Repo goc van nam nguyen o /mnt/c (chi co 1 ban de sua trong VSCode).

set -o pipefail

PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
VENV_DIR="$HOME/apc_env"
SRC_DIR="$PROJECT_DIR/apc/vision_modules/src"
BUILD_DIR="$HOME/build"
CUDA_SHIM="$HOME/cuda_shim"
ARCH="8.6"                 # RTX 3050 = Ampere, compute capability 8.6

step() { echo ""; echo "=============== $1 ==============="; }
ok()   { echo "STEP_OK: $1"; }
fail() { echo "STEP_FAIL: $1"; }

source "$VENV_DIR/bin/activate" || { fail "activate venv"; exit 1; }
export MAX_JOBS=2
echo "python=$(which python)  MAX_JOBS=$MAX_JOBS"

# --------------------------------------------------------------------- #
step "B1. Pin numpy 1.23.4"
pip install "numpy==1.23.4" >/dev/null 2>&1 && ok "numpy 1.23.4" || fail "numpy pin"

step "B2. Package thuan python (co wheel san)"
pip install python-box "trimesh==4.7.4" "open3d==0.19.0" "pyglet==1.5.27" \
    opencv-python scipy matplotlib "transformers==4.49.0" huggingface_hub timm \
    accelerate qwen-vl-utils pyyaml pytest \
    && ok "pure-python deps" || fail "pure-python deps"

step "B3. segment-anything"
if python -c "import segment_anything" 2>/dev/null; then
    ok "segment_anything da co"
else
    pip install "git+https://github.com/facebookresearch/segment-anything.git" \
        && ok "segment-anything" || fail "segment-anything"
fi

step "B4. depth_pro (pure python, cai non-editable tu $SRC_DIR)"
if python -c "import depth_pro" 2>/dev/null; then
    ok "depth_pro da co"
else
    ( cd "$SRC_DIR/ml-depth-pro" && pip install . ) \
        && ok "depth_pro" || fail "depth_pro"
fi

# --------------------------------------------------------------------- #
step "B5. Dung nvcc shim tu pip wheel (khong can sudo)"
pip install "nvidia-cuda-nvcc-cu12==12.4.131" "nvidia-cuda-cccl-cu12==12.4.127" \
    && ok "pip nvcc wheels" || fail "pip nvcc wheels"

NV_DIR="$(python -c 'import site,os; print(os.path.join(site.getsitepackages()[0], "nvidia"))')"
echo "nvidia pip dir: $NV_DIR"
rm -rf "$CUDA_SHIM"
mkdir -p "$CUDA_SHIM/bin" "$CUDA_SHIM/include" "$CUDA_SHIM/lib64" "$CUDA_SHIM/nvvm/bin" "$CUDA_SHIM/nvvm/libdevice"

# Gop bin/include/lib cua tat ca wheel nvidia_* vao 1 cay thu muc giong CUDA toolkit that
for d in "$NV_DIR"/*/; do
    [ -d "$d/bin" ]       && ln -sf "$d"bin/*       "$CUDA_SHIM/bin/"     2>/dev/null
    [ -d "$d/include" ]   && ln -sf "$d"include/*   "$CUDA_SHIM/include/" 2>/dev/null
    [ -d "$d/lib" ]       && ln -sf "$d"lib/*       "$CUDA_SHIM/lib64/"   2>/dev/null
    [ -d "$d/nvvm/bin" ]  && ln -sf "$d"nvvm/bin/*  "$CUDA_SHIM/nvvm/bin/" 2>/dev/null
    [ -d "$d/nvvm/libdevice" ] && ln -sf "$d"nvvm/libdevice/* "$CUDA_SHIM/nvvm/libdevice/" 2>/dev/null
done

export CUDA_HOME="$CUDA_SHIM"
export PATH="$CUDA_SHIM/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="$ARCH"
if nvcc --version >/dev/null 2>&1; then
    nvcc --version | tail -2
    ok "nvcc kha dung: $(nvcc --version | grep release | sed 's/.*release //')"
else
    fail "nvcc van khong chay duoc -> se phai cai CUDA toolkit bang sudo apt"
fi

step "B6. GroundingDINO (build CUDA ops, tu ban copy o $BUILD_DIR)"
if python -c "from groundingdino import _C" 2>/dev/null; then
    ok "groundingdino._C da co"
else
    mkdir -p "$BUILD_DIR"
    rm -rf "$BUILD_DIR/GroundingDINO"
    cp -r "$SRC_DIR/GroundingDINO" "$BUILD_DIR/GroundingDINO" && echo "copied to $BUILD_DIR/GroundingDINO"
    ( cd "$BUILD_DIR/GroundingDINO" && pip install . --no-build-isolation ) \
        && ok "groundingdino installed" || fail "groundingdino install"
    if python -c "from groundingdino import _C" 2>/dev/null; then
        ok "groundingdino._C import duoc (CUDA ops OK)"
    else
        fail "groundingdino._C KHONG import duoc -> detection phai chay o CPU"
    fi
fi

# --------------------------------------------------------------------- #
step "B7. detectron2 (CPU-only build)"
if python -c "import detectron2" 2>/dev/null; then
    ok "detectron2 da co"
else
    ( unset CUDA_HOME; export FORCE_CUDA=0; \
      pip install "git+https://github.com/facebookresearch/detectron2.git" --no-build-isolation ) \
        && ok "detectron2" || fail "detectron2"
fi

step "B8. pytorch3d v0.7.8 (CPU-only build, co the 30-60 phut voi MAX_JOBS=2)"
if python -c "import pytorch3d" 2>/dev/null; then
    ok "pytorch3d da co"
else
    ( unset CUDA_HOME; export FORCE_CUDA=0; \
      pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8" --no-build-isolation ) \
        && ok "pytorch3d" || fail "pytorch3d"
fi

step "B9. Re-pin numpy (cac buoc tren co the nang cap numpy)"
pip install "numpy==1.23.4" >/dev/null 2>&1 && ok "numpy re-pinned" || fail "numpy re-pin"

# --------------------------------------------------------------------- #
step "B10. Import check"
cd "$PROJECT_DIR"
python - <<'PY'
import importlib
for m in ["numpy", "scipy", "torch", "torchvision", "cv2", "trimesh", "open3d", "pyglet",
          "matplotlib", "box", "transformers", "segment_anything", "groundingdino",
          "depth_pro", "detectron2", "pytorch3d"]:
    try:
        mod = importlib.import_module(m)
        print(f"IMPORT_OK   {m:18s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"IMPORT_FAIL {m:18s} {type(e).__name__}: {e}")

try:
    from groundingdino import _C
    print("IMPORT_OK   groundingdino._C   (CUDA ops da build)")
except Exception as e:
    print(f"IMPORT_FAIL groundingdino._C   {type(e).__name__}: {e}")

import torch
print("torch", torch.__version__, "| cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | VRAM {p.total_memory/1024**3:.2f} GB | sm_{p.major}{p.minor}")
PY

step "B11. Import check code APC (vision modules, KHONG load VLM)"
python - <<'PY'
import sys
for p in ["apc/vision_modules", "apc/vision_modules/src/omni3d",
          "apc/vision_modules/src/orient_anything", "apc/vision_modules/src/GroundingDINO"]:
    sys.path.append(p)
for mod in ["apc.vision_modules.vision_utils", "apc.renderer", "apc.vision_modules.depth",
            "apc.vision_modules.detection", "apc.vision_modules.orientation"]:
    try:
        __import__(mod)
        print(f"APC_IMPORT_OK   {mod}")
    except Exception as e:
        print(f"APC_IMPORT_FAIL {mod}  {type(e).__name__}: {e}")
PY

step "B12. Test OpenGL cho trimesh (WSLg DISPLAY=:0)"
python - <<'PY'
import os
print("DISPLAY =", os.environ.get("DISPLAY"))
try:
    import trimesh
    s = trimesh.Scene([trimesh.creation.box()])
    png = s.save_image(resolution=(256, 256))
    open("/tmp/trimesh_test.png", "wb").write(png)
    print(f"RENDER_OK  trimesh.save_image -> {len(png)} bytes (/tmp/trimesh_test.png)")
except Exception as e:
    print(f"RENDER_FAIL {type(e).__name__}: {e}  -> can 'sudo apt install xvfb' va dung xvfb-run")
PY

step "B13. Ghi test/ENV_VERSIONS.md"
{
    echo "# Phien ban moi truong (tu dong sinh boi test/setup_wsl_phaseB.sh)"
    echo ""
    echo "Ngay: $(date -Iseconds)"
    echo "Host: WSL2 Ubuntu-22.04 | GPU: RTX 3050 6GB Laptop | venv: $VENV_DIR"
    echo ""
    echo "## Commit cac repo con"
    echo '```'
    for d in GroundingDINO ml-depth-pro orient_anything omni3d; do
        printf "%-16s %s\n" "$d" "$(git -C "$SRC_DIR/$d" rev-parse HEAD 2>/dev/null || echo MISSING)"
    done
    echo '```'
    echo ""
    echo "## pip freeze"
    echo '```'
    pip freeze
    echo '```'
} > "$PROJECT_DIR/test/ENV_VERSIONS.md"
ok "wrote test/ENV_VERSIONS.md"

echo ""
echo "PHASE_B_DONE"
