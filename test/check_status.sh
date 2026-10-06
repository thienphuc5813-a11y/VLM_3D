#!/usr/bin/env bash
# Kiem tra nhanh trang thai Phase A / Phase B / tai checkpoint.
# Chay: wsl -d Ubuntu-22.04 -- bash "/mnt/c/.../test/check_status.sh"
# hoac tu trong Ubuntu: bash test/check_status.sh   (neu dang o goc repo trong WSL)

CKPT="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051/apc/vision_modules/src/checkpoints"

echo "========================================"
echo "CHECKPOINT (ky vong: sam ~2.4G, groundingdino ~662M, depth_pro ~1.8G)"
echo "========================================"
ls -lh "$CKPT" 2>/dev/null || echo "(chua co thu muc checkpoints)"

echo ""
echo "========================================"
echo "TAI XUONG DANG CHAY (wget)"
echo "========================================"
if pgrep -af wget > /dev/null 2>&1; then
    pgrep -af wget
else
    echo "(khong co wget nao dang chay)"
fi

echo ""
echo "========================================"
echo "PHASE A (clone repo + tai checkpoint)"
echo "========================================"
if [ -f ~/apc_setup_phaseA.log ]; then
    grep -E "STEP_OK|STEP_FAIL|PHASE_A_DONE" ~/apc_setup_phaseA.log
    if pgrep -f phaseA.sh > /dev/null; then echo ">>> A_STATUS: DANG CHAY"; else echo ">>> A_STATUS: DA DUNG (xem PHASE_A_DONE o tren co chua)"; fi
else
    echo "(chua co log)"
fi

echo ""
echo "========================================"
echo "PHASE B (cai package + build extension)"
echo "========================================"
if [ -f ~/apc_setup_phaseB.log ]; then
    grep -E "STEP_OK|STEP_FAIL|PHASE_B_DONE" ~/apc_setup_phaseB.log
    echo "--- dong cuoi (dang lam gi) ---"
    tail -5 ~/apc_setup_phaseB.log
    if pgrep -f phaseB.sh > /dev/null; then echo ">>> B_STATUS: DANG CHAY"; else echo ">>> B_STATUS: DA DUNG (xem PHASE_B_DONE o tren co chua)"; fi
else
    echo "(chua co log)"
fi
