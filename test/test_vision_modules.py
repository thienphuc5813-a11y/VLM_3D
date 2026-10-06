'''
LOCAL_RUN_PLAN.md muc 4.3: chay tung vision module that tren demo/sample_image_man.jpg, do VRAM.
Moi test nen chay trong mot process rieng (xem test/run_vision_tests.sh), de do VRAM sach.

- GroundingDINO chay CPU vi may nay khong co _C (build CPU-only); B5 da sua nen
  DetectionModule(config, "cpu") tu truyen device="cpu" vao predict().
- run_detection tra ve box cxcywh chuan hoa (dung thiet ke); test tu doi sang xyxy pixel
  truoc khi goi run_segmentation, giong detection.py:118-126.

OOM khong phai loi (muc 4.3 buoc 4): ghi vao test/outputs/vram.md roi SKIP.
'''
import os
import time

import numpy as np
import pytest
import torch
import yaml
from box import Box
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "test", "outputs", "vision")
VRAM_MD = os.path.join(REPO_ROOT, "test", "outputs", "vram.md")
DEMO_IMAGE = os.path.join(REPO_ROOT, "demo", "sample_image_man.jpg")
PERSON_BOX = [135, 105, 285, 330]   # xyxy pixel, chon tay tu anh demo 640x427
GPU = "cuda"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="can GPU")


@pytest.fixture(scope="module")
def config():
    with open(os.path.join(REPO_ROOT, "test", "configs", "no_vlm.yaml")) as f:
        return Box(yaml.safe_load(f))


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    # Duong dan checkpoint trong config la tuong doi; vision_tower.py:107 dung cache_dir='./'
    monkeypatch.chdir(REPO_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@pytest.fixture(scope="module")
def image():
    return Image.open(DEMO_IMAGE).convert("RGB")


def _gb(x):
    return x / 1024 ** 3


def record_vram(name, note=""):
    free, total = torch.cuda.mem_get_info()
    row = (f"| {name} | {_gb(torch.cuda.max_memory_allocated()):.2f} | "
           f"{_gb(torch.cuda.max_memory_reserved()):.2f} | {_gb(free):.2f} / {_gb(total):.2f} | {note} |\n")
    new_file = not os.path.exists(VRAM_MD)
    with open(VRAM_MD, "a") as f:
        if new_file:
            f.write("| Buoc | Peak allocated (GB) | Peak reserved (GB) | Free/Total sau buoc (GB) | Ghi chu |\n")
            f.write("|---|---|---|---|---|\n")
        f.write(row)
    print("VRAM", row.strip())


def run_or_record_oom(name, fn):
    try:
        return fn()
    except torch.cuda.OutOfMemoryError as e:
        record_vram(name, f"OOM: {str(e).splitlines()[0][:80]}")
        pytest.skip(f"OOM o buoc {name} (ghi nhan, khong phai loi)")


def _cxcywh_to_xyxy_pixel(box, W, H):
    cx, cy, w, h = [float(v) for v in box]
    return [int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)]


def _make_detection_module(config, monkeypatch):
    import apc.vision_modules.detection as det_mod
    from segment_anything import SamPredictor

    det = det_mod.DetectionModule(config, "cpu")
    det.segmentation_model.to(GPU)
    det.segmentation_predictor = SamPredictor(det.segmentation_model)
    return det


# ===================================================================== #
def test_orientation_module(config, image):
    from apc.vision_modules.orientation import OrientationModule

    def run():
        om = OrientationModule(config, GPU)
        record_vram("orientation: load", "Orient-Anything (DINOv2-L)")
        vec = om.run_orientation_estimation(
            image=image, category="person", bbox=PERSON_BOX,
            trace_save_dir=OUT_DIR, visualize_trace=False,
        )
        angles = om.get_3angle(image.crop(PERSON_BOX))
        return vec, angles

    vec, angles = run_or_record_oom("orientation", run)
    record_vram("orientation: infer")
    print(f"azimuth={angles[0]:.0f} polar={angles[1]:.0f} rotation={angles[2]:.0f} -> direction={np.round(vec, 3)}")
    assert np.all(np.isfinite(vec))
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-3)


def test_detection_module(config, image, monkeypatch):
    W, H = image.size

    def run():
        det = _make_detection_module(config, monkeypatch)
        record_vram("detection: load", "GDINO o CPU (khong co _C), SAM ViT-H o GPU")
        _, image_processed = det.detection_process_image(image)
        t0 = time.time()
        boxes = det.run_detection(image_processed, "person")
        t_gdino = time.time() - t0
        assert len(boxes) > 0, "GroundingDINO khong tim thay 'person'"
        box_xyxy = _cxcywh_to_xyxy_pixel(boxes[0], W, H)
        masks = det.run_segmentation(image, box_xyxy)
        return boxes, box_xyxy, masks, t_gdino

    boxes, box_xyxy, masks, t_gdino = run_or_record_oom("detection", run)
    record_vram("detection: infer", f"GDINO CPU {t_gdino:.1f}s, anh {W}x{H} full-res")

    cxcywh = np.array([[float(v) for v in b] for b in boxes])
    assert np.all((cxcywh >= 0) & (cxcywh <= 1)), "box khong o dang cxcywh chuan hoa nhu B2 mo ta"

    mask = masks[2] > 0.5
    assert mask.sum() > 0
    x0, y0, x1, y1 = box_xyxy
    inside = mask[max(y0, 0):y1, max(x0, 0):x1].sum() / mask.sum()
    assert inside > 0.9, f"chi {inside:.0%} mask nam trong box"

    overlay = np.array(image).astype(np.float32)
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])
    Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(OUT_DIR, "detection_person_mask.png"))
    print(f"box xyxy={box_xyxy} (thu cong chon {PERSON_BOX}), mask {mask.sum()} px, {inside:.0%} trong box")


def test_depth_module(config, image):
    from apc.vision_modules.depth import DepthModule

    def run():
        dm = DepthModule(config, GPU)
        record_vram("depth: load", "Depth Pro fp16")
        with torch.no_grad():
            return dm.run_depth_estimation(image)

    depth = run_or_record_oom("depth", run)
    record_vram("depth: infer")

    W, H = image.size
    assert depth.shape == (H, W)
    assert np.all(np.isfinite(depth)) and depth.min() > 0
    x0, y0, x1, y1 = PERSON_BOX
    person_depth = float(np.median(depth[y0:y1, x0:x1]))
    print(f"depth min={depth.min():.2f} median={np.median(depth):.2f} max={depth.max():.2f} | nguoi ~{person_depth:.2f} m")
    assert 0.5 < person_depth < 15, f"depth cua nguoi {person_depth:.2f} m khong hop ly"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(os.path.join(OUT_DIR, "depth.png"), depth, cmap="turbo")


def test_all_three_together(config, image, monkeypatch):
    from apc.vision_modules.depth import DepthModule
    from apc.vision_modules.orientation import OrientationModule

    def run():
        det = _make_detection_module(config, monkeypatch)
        dm = DepthModule(config, GPU)
        om = OrientationModule(config, GPU)
        record_vram("ca 3: load", "GDINO CPU + SAM/Depth/Orient GPU")
        with torch.no_grad():
            dm.run_depth_estimation(image)
            det.run_segmentation(image, PERSON_BOX)
            om.run_orientation_estimation(image=image, category="person", bbox=PERSON_BOX,
                                          trace_save_dir=OUT_DIR, visualize_trace=False)

    run_or_record_oom("ca 3 cung luc", run)
    record_vram("ca 3: infer")
