#!/usr/bin/env bash
# Chay pytest trong venv WSL tu goc repo. Vi du:
#   wsl -d Ubuntu-22.04 -- bash "/mnt/c/.../test/run_pytest.sh" test/test_geometry.py -v
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
source "$HOME/apc_env/bin/activate"
# Giao thuc Xet cua HuggingFace bi treo trong WSL2 tren may nay; dung HTTPS thuong
export HF_HUB_DISABLE_XET=1
cd "$PROJECT_DIR" || exit 1
python -m pytest -p no:cacheprovider "$@"
