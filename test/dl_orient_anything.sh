#!/usr/bin/env bash
# LOCAL_RUN_PLAN 1.8: tai checkpoint Orient-Anything + processor dinov2-large vao apc/vision_modules/src/checkpoints
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
source "$HOME/apc_env/bin/activate"
cd "$PROJECT_DIR" || exit 1
export HF_HUB_ENABLE_HF_TRANSFER=0

python setup/download_orient_anything.py --cache_dir apc/vision_modules/src/checkpoints \
    && echo "STEP_OK: orient-anything weight" || echo "STEP_FAIL: orient-anything weight"

# Processor: orientation.py:50-53 dung cache_dir=checkpoints.
# Trong so DINOv2: vision_tower.py:107 dung cache_dir='./' (thu muc dang dung = goc repo).
python - <<'PY' && echo "STEP_OK: dinov2-large processor + model" || echo "STEP_FAIL: dinov2-large"
from transformers import AutoImageProcessor, AutoModel
AutoImageProcessor.from_pretrained("facebook/dinov2-large", cache_dir="apc/vision_modules/src/checkpoints")
AutoModel.from_pretrained("facebook/dinov2-large", cache_dir="./")
PY

ls -la apc/vision_modules/src/checkpoints
echo "DL_ORIENT_DONE"
