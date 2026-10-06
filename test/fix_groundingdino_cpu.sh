#!/usr/bin/env bash
# Build lai GroundingDINO o CPU-only (khong can nvcc/sudo), vi pip wheel nvidia-cuda-nvcc-cu12
# chi co ptxas + header, KHONG co binary nvcc thuc su -> khong du de build CUDA extension.
#
# He qua: groundingdino._C se khong ton tai. Theo ms_deform_attn.py:330, nhanh pure-PyTorch
# chi chay khi tensor o CPU, nen DetectionModule phai duoc goi voi device="cpu" trong moi
# script test/ (apc/vision_modules/detection.py khong sua gi, chi doi device luc goi).

set -o pipefail
BUILD_DIR="$HOME/build/GroundingDINO"
VENV_DIR="$HOME/apc_env"

source "$VENV_DIR/bin/activate" || { echo "FAIL: activate venv"; exit 1; }
unset CUDA_HOME
export FORCE_CUDA=0
export MAX_JOBS=2

echo "=== Build GroundingDINO (CPU-only) tu $BUILD_DIR ==="
cd "$BUILD_DIR" || { echo "FAIL: build dir missing"; exit 1; }
rm -rf build *.egg-info
pip install . --no-build-isolation
INSTALL_EXIT=$?
echo "pip install exit code: $INSTALL_EXIT"

echo ""
echo "=== Kiem tra import ==="
python - <<'PY'
try:
    import groundingdino
    print("IMPORT_OK groundingdino", getattr(groundingdino, "__version__", "?"))
except Exception as e:
    print(f"IMPORT_FAIL groundingdino {type(e).__name__}: {e}")

try:
    from groundingdino import _C
    print("IMPORT_OK groundingdino._C (khong ngo, CUDA ops co san?)")
except Exception as e:
    print(f"IMPORT_EXPECTED_FAIL groundingdino._C (binh thuong, da biet truoc): {type(e).__name__}: {e}")

try:
    from groundingdino.util.inference import load_model
    print("IMPORT_OK groundingdino.util.inference.load_model")
except Exception as e:
    print(f"IMPORT_FAIL groundingdino.util.inference {type(e).__name__}: {e}")
PY

echo "FIX_GDINO_DONE"
