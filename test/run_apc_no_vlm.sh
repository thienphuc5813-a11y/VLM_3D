#!/usr/bin/env bash
# Chay test/run_apc_no_vlm.py trong venv WSL. Vi du:
#   wsl -d Ubuntu-22.04 -- bash "/mnt/c/.../test/run_apc_no_vlm.sh" --case man
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
source "$HOME/apc_env/bin/activate"
export HF_HUB_DISABLE_XET=1
cd "$PROJECT_DIR" || exit 1
mkdir -p test/outputs
# DepthPro in nguyen kien truc mang ra log => ghi log day du ra file, chi in dong quan trong
python -W ignore test/run_apc_no_vlm.py "$@" > test/outputs/run_apc_no_vlm.log 2>&1
grep -E "^#{8}|Error|error|Traceback|loi|Reference viewer|No detection" test/outputs/run_apc_no_vlm.log | head -40
for d in test/outputs/*/; do [ -f "$d/run_log.json" ] && echo "--- $d" && cat "$d/run_log.json"; done
