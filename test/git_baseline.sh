#!/usr/bin/env bash
# LOCAL_RUN_PLAN 1.2: git init + commit baseline (bat buoc truoc khi sua apc/ o buoc 6).
# Commit 1: code upstream nguyen ban. Commit 2: tai lieu + test harness (chua sua apc/).
PROJECT_DIR="/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051"
cd "$PROJECT_DIR" || exit 1

if [ -d .git ]; then echo "da co .git, dung lai"; exit 1; fi
git init -q -b main
git config core.fileMode false      # /mnt/c hien moi file la 777
git config core.autocrlf false
git config user.name "thien"
git config user.email "nckhfpt123@gmail.com"

git add LICENSE README.md run_APC.py run_APC.ipynb apc assets demo setup
git commit -q -m "baseline: upstream APC-VLM (chua sua gi)"

git add .gitignore APC_project_context.md claude test
git commit -q -m "test: harness theo LOCAL_RUN_PLAN muc 4 + tai lieu phan tich (chua sua apc/)"

git log --oneline
echo "--- file khong duoc track (phai la rong hoac chi la file chu y bo qua) ---"
git status --short
echo "--- kich thuoc .git ---"
du -sh .git
