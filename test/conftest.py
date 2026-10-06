'''
Pytest conftest cho test/: them sys.path giong apc_pipeline.py:15-17 va run_APC.py:18,
de 'import apc...' hoat dong khi chay pytest tu goc repo. Khong sua gi trong apc/.
'''
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# append (khong insert) de khop dung apc_pipeline.py:15-17 / run_APC.py:18,
# va khong che package da pip install trong venv
for p in [
    os.path.join(REPO_ROOT, "apc/vision_modules"),
    os.path.join(REPO_ROOT, "apc/vision_modules/src/omni3d"),
    os.path.join(REPO_ROOT, "apc/vision_modules/src/orient_anything"),
    os.path.join(REPO_ROOT, "apc/vision_modules/src/GroundingDINO"),
]:
    if p not in sys.path:
        sys.path.append(p)

OUTPUT_DIR = os.path.join(REPO_ROOT, "test", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
