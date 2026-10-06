#!/usr/bin/env bash
# LOCAL_RUN_PLAN muc 6: chay bo test nhanh, CHI commit neu pass.
# Bug da sua ma quen bo xfail => strict XPASS => fail => khong commit.
#   wsl -d Ubuntu-22.04 -- bash "/mnt/c/.../test/commit_step.sh" "fix(B1): ..."
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
MSG="$1"
[ -z "$MSG" ] && { echo "can commit message"; exit 2; }
source "$HOME/apc_env/bin/activate"
export HF_HUB_DISABLE_XET=1
cd "$PROJECT_DIR" || exit 1

python -m pytest -p no:cacheprovider -q -rxX --tb=short -W ignore \
    test/test_geometry.py test/test_renderer.py test/test_pipeline_bugs.py \
    $( [ -f test/test_options.py ] && echo test/test_options.py ) 2>&1 | tail -25
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "PYTEST_FAILED => KHONG commit"
    exit 1
fi

git add -A
git commit -q -m "$MSG" && git log --oneline -1 && git show --stat --format= HEAD
