#!/usr/bin/env bash
# LOCAL_RUN_PLAN buoc 1.9 (render OpenGL), 1.10 (smoke import), ghi test/ENV_VERSIONS.md
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
SRC="$PROJECT_DIR/apc/vision_modules/src"
source "$HOME/apc_env/bin/activate"
cd "$PROJECT_DIR" || exit 1

echo "=== 1.9 OpenGL render (DISPLAY=$DISPLAY) ==="
python -c "import trimesh; s=trimesh.Scene([trimesh.creation.box()]); png=s.save_image(resolution=(256,256)); open('/tmp/t.png','wb').write(png); print('RENDER_OK', len(png), 'bytes')" 2>&1 | tail -3

echo ""
echo "=== 1.10 smoke import ==="
python - <<'PY' 2>&1 | grep -E "IMPORT_|Error" | head -20
import sys
for p in ['apc/vision_modules/src/omni3d', 'apc/vision_modules/src/orient_anything', 'apc/vision_modules/src/GroundingDINO']:
    sys.path.append(p)
for m in ["apc.vision_modules", "apc.renderer", "apc.apc_pipeline"]:
    try:
        __import__(m)
        print("IMPORT_OK  ", m)
    except Exception as e:
        print("IMPORT_FAIL", m, type(e).__name__, e)
PY

echo ""
echo "=== ghi test/ENV_VERSIONS.md ==="
{
    echo "# Phien ban moi truong"
    echo ""
    echo "Sinh tu dong boi test/finish_step1.sh luc $(date -Iseconds)."
    echo "WSL2 Ubuntu-22.04, venv \`~/apc_env\`, GPU RTX 3050 6GB Laptop, driver 566.03."
    echo ""
    echo "## Lech so voi LOCAL_RUN_PLAN.md"
    echo ""
    echo "- Repo giu nguyen o \`/mnt/c/...\` (khong copy sang \`~/apc-vlm\`), venv o \`~/apc_env\` (khong dung conda)."
    echo "- Khong co \`nvcc\` (pip wheel \`nvidia-cuda-nvcc-cu12\` chi co ptxas, khong co nvcc)."
    echo "  GroundingDINO, detectron2, pytorch3d deu build CPU-only."
    echo "  GroundingDINO khong co \`_C\` => theo ms_deform_attn.py:330, DetectionModule phai chay o device=cpu."
    echo "- numpy 1.23.4 (theo setup/requirements.txt). scipy 1.15.3 bao can >=1.23.5 nhung import duoc."
    echo "  Trong apc/ chi co omni3d_evaluation.py dung np.float (APC khong goi), nen co the nang len 1.26.4 neu can."
    echo ""
    echo "## Commit cac repo con"
    echo ""
    echo '```'
    for d in GroundingDINO ml-depth-pro orient_anything omni3d; do
        printf "%-16s %s\n" "$d" "$(git -C "$SRC/$d" rev-parse HEAD 2>/dev/null || echo MISSING)"
    done
    echo "pytorch3d        V0.7.8 (75ebeeaea0908c5527e7b1e305fbc7681382db47)"
    echo '```'
    echo ""
    echo "## pip freeze"
    echo ""
    echo '```'
    pip freeze
    echo '```'
} > "$PROJECT_DIR/test/ENV_VERSIONS.md"
echo "wrote test/ENV_VERSIONS.md ($(wc -l < "$PROJECT_DIR/test/ENV_VERSIONS.md") dong)"
