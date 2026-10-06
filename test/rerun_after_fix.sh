#!/usr/bin/env bash
# LOCAL_RUN_PLAN muc 6 (cuoi): luu ket qua cu vao before_fix/, chay lai 4.4 va 4.3 sau khi sua, so sanh.
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
source "$HOME/apc_env/bin/activate"
export HF_HUB_DISABLE_XET=1
cd "$PROJECT_DIR" || exit 1

if [ ! -d test/outputs/before_fix ]; then
    mkdir -p test/outputs/before_fix
    for d in man dog_chair spiderman woman_chair vision vram.md; do
        [ -e "test/outputs/$d" ] && mv "test/outputs/$d" test/outputs/before_fix/
    done
fi

echo "################ 4.4 run_apc_no_vlm (sau sua) ################"
python -W ignore test/run_apc_no_vlm.py > test/outputs/run_apc_no_vlm.log 2>&1
grep -E "^#{8}|Traceback|loi:" test/outputs/run_apc_no_vlm.log

echo ""
echo "################ 4.3 vision modules (sau sua) ################"
bash test/run_vision_tests.sh

echo ""
echo "################ so sanh truoc / sau ################"
python test/compare_before_after.py
echo "RERUN_AFTER_FIX_DONE"
