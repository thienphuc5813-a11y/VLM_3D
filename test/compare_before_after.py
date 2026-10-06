'''
LOCAL_RUN_PLAN muc 6 (cuoi): so ket qua muc 4.4 truoc va sau khi sua B1-B9.
Truoc: test/outputs/before_fix/<case>/   Sau: test/outputs/<case>/
Ghi bang ket qua ra test/outputs/before_after.md

Chay tu goc repo:  python test/compare_before_after.py
'''
import json
import os

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "test", "outputs")
BEFORE = os.path.join(OUT, "before_fix")
CASES = ["man", "dog_chair", "spiderman", "woman_chair"]


def load(path):
    with open(path) as f:
        return json.load(f)


def max_scene_diff(a, b):
    diffs = []
    for name in set(a) | set(b):
        if name not in a or name not in b:
            return float("inf"), f"vat '{name}' chi co o mot ben"
        for key in ("position", "orientation"):
            diffs.append(np.abs(np.array(a[name][key], float) - np.array(b[name][key], float)).max())
    return float(max(diffs)), ""


def image_diff(p1, p2):
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return None
    a = np.asarray(Image.open(p1).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(p2).convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).mean())


def sides(log):
    return {k: v["side"] + "/" + v["front_or_back"].split(" ")[0]
            for k, v in log.get("viewer_frame_summary", {}).items()}


rows = ["| Anh | Lech toa do max (he camera) | Lech toa do max (he viewer) | Trai/phai giong nhau? "
        "| Lech pixel TB visual_prompt | Thoi gian truoc -> sau (s) |",
        "|---|---|---|---|---|---|"]
for case in CASES:
    b_dir, a_dir = os.path.join(BEFORE, case), os.path.join(OUT, case)
    try:
        b_log, a_log = load(os.path.join(b_dir, "run_log.json")), load(os.path.join(a_dir, "run_log.json"))
        d_cam, n1 = max_scene_diff(load(os.path.join(b_dir, "abstract_camera.json")),
                                   load(os.path.join(a_dir, "abstract_camera.json")))
        d_ref, n2 = max_scene_diff(load(os.path.join(b_dir, "abstract_ref.json")),
                                   load(os.path.join(a_dir, "abstract_ref.json")))
        same_sides = sides(b_log) == sides(a_log)
        px = image_diff(os.path.join(b_dir, "visual_prompt.png"), os.path.join(a_dir, "visual_prompt.png"))
        rows.append(f"| {case} | {d_cam:.4f} {n1} | {d_ref:.4f} {n2} | {'co' if same_sides else 'KHONG: ' + str(sides(a_log))} "
                    f"| {px:.2f} | {b_log['seconds']} -> {a_log['seconds']} |")
    except FileNotFoundError as e:
        rows.append(f"| {case} | thieu file: {os.path.relpath(e.filename, REPO_ROOT)} | | | | |")

table = "\n".join(rows)
print(table)
with open(os.path.join(OUT, "before_after.md"), "w") as f:
    f.write("# Muc 4.4 truoc va sau khi sua B1-B9\n\n" + table + "\n")
