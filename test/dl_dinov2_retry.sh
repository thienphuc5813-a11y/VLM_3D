#!/usr/bin/env bash
# Tai DINOv2-large vao ./ (goc repo, theo vision_tower.py:107) voi retry.
# Mang trong WSL2 hay treo ket noi dai: moi lan thu toi da 120s, HF hub tu resume file .incomplete.
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
source "$HOME/apc_env/bin/activate"
cd "$PROJECT_DIR" || exit 1

pkill -f dl_orient_anything.sh 2>/dev/null
pkill -f "AutoModel.from_pretrained" 2>/dev/null
pkill -f "dinov2-large" 2>/dev/null
sleep 1

# Giao thuc Xet (hf-xet) bi dung im trong WSL2 tren may nay; HTTPS thuong thi chay ~3.7MB/s
export HF_HUB_DISABLE_XET=1

for attempt in $(seq 1 30); do
    timeout 300 python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/dinov2-large', cache_dir='./', allow_patterns=['*.json', '*.safetensors'])
print('DINOV2_OK')
" 2>&1 | grep -E "DINOV2_OK|Error" && break
    echo "attempt $attempt: chua xong, size=$(du -sh models--facebook--dinov2-large 2>/dev/null | cut -f1)"
done

du -sh models--facebook--dinov2-large
echo "DL_DINOV2_DONE"
