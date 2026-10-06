#!/usr/bin/env bash
# Kiem tra nhanh moi truong: package nao import duoc, numpy version, build pytorch3d con chay khong.
# Chay: wsl -d Ubuntu-22.04 -- bash "/mnt/c/.../test/check_env.sh"
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
source "$HOME/apc_env/bin/activate"
cd "$PROJECT_DIR" || exit 1

echo "=== build pytorch3d con chay? ==="
pgrep -af "pip install" || echo "(khong co pip nao dang chay)"
echo "--- duoi log retry_p3d ---"
tail -4 ~/retry_p3d.log 2>/dev/null

echo ""
echo "=== import check ==="
python - <<'PY' 2>&1 | grep -v -E "Warning|warn"
import importlib
for m in ["numpy", "scipy", "cv2", "pandas", "torch", "trimesh", "open3d", "pyglet",
          "transformers", "segment_anything", "groundingdino", "depth_pro",
          "detectron2", "pytorch3d"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK    {m:18s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"FAIL  {m:18s} {type(e).__name__}: {str(e)[:120]}")
import torch
print("cuda:", torch.cuda.is_available())
PY

echo ""
echo "=== checkpoints ==="
ls -lh apc/vision_modules/src/checkpoints
