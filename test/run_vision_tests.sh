#!/usr/bin/env bash
# LOCAL_RUN_PLAN 4.3: chay tung test vision trong process rieng de do VRAM sach.
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
source "$HOME/apc_env/bin/activate"
# Giao thuc Xet cua HuggingFace bi treo trong WSL2 tren may nay; dung HTTPS thuong
export HF_HUB_DISABLE_XET=1
cd "$PROJECT_DIR" || exit 1

TESTS="${*:-test_orientation_module test_detection_module test_depth_module test_all_three_together}"
[ -z "$*" ] && rm -f test/outputs/vram.md

for t in $TESTS; do
    echo ""
    echo "################ $t ################"
    python -m pytest -p no:cacheprovider test/test_vision_modules.py -k "$t" -q -s -rs --tb=short -W ignore 2>&1 \
        | grep -E "VRAM|azimuth|box xyxy|depth min|PASS|FAIL|SKIP|Error|error|passed|failed|skipped|assert" | head -30
done

echo ""
echo "################ test/outputs/vram.md ################"
cat test/outputs/vram.md
