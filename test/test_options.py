'''
LOCAL_RUN_PLAN.md muc 6: option R1-R4 trong config. Mac dinh giu hanh vi goc
(cac test o test_geometry.py / test_pipeline_bugs.py chay voi mac dinh), o day kiem tra
khi BAT option thi hanh vi doi dung nhu mo ta.

Chay:  pytest test/test_options.py -v
'''
import numpy as np
import pytest
from box import Box
from PIL import Image

W, H = 200, 100


def _gray_image():
    return Image.fromarray(np.full((H, W, 3), 128, dtype=np.uint8))


def _rect_mask(u_c, v_c, half=10):
    mask = np.zeros((H, W), dtype=np.float32)
    mask[v_c - half: v_c + half + 1, u_c - half: u_c + half + 1] = 1.0
    return mask


def _depth_module(depth_cfg):
    mod = pytest.importorskip("apc.vision_modules.depth")
    dm = object.__new__(mod.DepthModule)     # bo qua __init__ (khong nap Depth Pro); meo cho test
    dm.config = Box({"depth": depth_cfg})
    return dm


# --------------------------------------------------------------------- #
# R1: depth.use_estimated_focal
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg, expected_focal", [({}, 4.0 * W), ({"use_estimated_focal": False}, 4.0 * W),
                                                 ({"use_estimated_focal": True}, 400.0)])
def test_R1_focal_length_option(cfg, expected_focal):
    dm = _depth_module(cfg)
    dm.last_focal_px = 400.0     # gia tri Depth Pro uoc luong (run_depth_estimation luu lai)
    d, u, v = 2.0, 150, 30
    depth = np.full((H, W), 10.0, dtype=np.float32)
    mask = _rect_mask(u, v)
    depth[mask > 0.5] = d
    pos, _ = dm.unproject_to_3D(image=_gray_image(), depth=depth, segment_mask=mask, return_box3D=False)
    expected = [d * (u - W / 2) / expected_focal, d * (v - H / 2) / expected_focal, d]
    np.testing.assert_allclose(pos, expected, atol=1e-2)


# --------------------------------------------------------------------- #
# R2: depth.mode_filter_ratio
# --------------------------------------------------------------------- #
def _ramp_depth():
    depth = np.full((H, W), 10.0, dtype=np.float32)
    mask = _rect_mask(100, 50, half=20)
    ys, xs = np.where(mask > 0.5)
    depth[ys, xs] = 2.0 + (xs - xs.min()) / (xs.max() - xs.min())   # doc tu 2.0 -> 3.0
    return depth, mask


@pytest.mark.parametrize("cfg, z_low, z_high", [
    ({}, 2.0, 2.25),                          # mac dinh 0.1: bi keo ve phia camera (R2)
    ({"mode_filter_ratio": 0.1}, 2.0, 2.25),
    ({"mode_filter_ratio": 0}, 2.45, 2.55),   # tat loc: trung vi cua ca be mat ~2.5
])
def test_R2_mode_filter_ratio_option(cfg, z_low, z_high):
    dm = _depth_module(cfg)
    depth, mask = _ramp_depth()
    pos, _ = dm.unproject_to_3D(image=_gray_image(), depth=depth, segment_mask=mask, return_box3D=False)
    assert z_low <= pos[2] <= z_high, f"z={pos[2]:.3f}"


# --------------------------------------------------------------------- #
# R3: vlm.response_role
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("vlm_cfg, expected_role", [({}, "system"), ({"response_role": "system"}, "system"),
                                                    ({"response_role": "assistant"}, "assistant")])
def test_R3_response_role_option(tmp_path, vlm_cfg, expected_role):
    import test_pipeline_bugs as bugs     # dung lai stub (VLM, render, detection) cua file do
    config = bugs._config()
    config.vlm.update(vlm_cfg)
    vlm = bugs.StubVLM()
    apc = bugs._make_apc(config, vlm=vlm)
    apc.do_perspective_prompting_visual("person", bugs._scene_ref(), "is obj0 left of obj1?",
                                        **bugs._apc_args(tmp_path))
    # Cuoc goi VLM thu 3 (abstract -> real) nhan lai cau tra loi cua chinh VLM o luot truoc
    roles = [m["role"] for m in vlm.calls[2]]
    assert roles == ["user", expected_role, "user"]


# --------------------------------------------------------------------- #
# R4: detection.refinement_parse
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("det_cfg, expected_idx", [
    ({}, 1),                                     # goc: quet i=0..4, '1' xuat hien truoc => 1
    ({"refinement_parse": "substring"}, 1),
    ({"refinement_parse": "first_integer"}, 3),  # so dau tien trong cau tra loi
])
def test_R4_refinement_parse_option(det_cfg, expected_idx):
    import matplotlib
    matplotlib.use("Agg")
    import torch
    import test_pipeline_bugs as bugs
    det_mod = pytest.importorskip("apc.vision_modules.detection")

    det = object.__new__(det_mod.DetectionModule)    # bo qua __init__ (khong nap model); meo cho test
    det.config = Box({"detection": det_cfg})
    det.num_candidates = 5
    boxes = [torch.tensor([0.1 + 0.18 * i, 0.5, 0.1, 0.4]) for i in range(5)]   # cxcywh chuan hoa
    vlm = bugs.StubVLM(["Image 3 is the best match, not 1."])

    box = det.run_detection_refinement(vlm, _gray_image(), "chair", boxes_output=boxes, trace_save_dir=None)

    cx, cy, w, h = [float(v) for v in boxes[expected_idx]]
    expected = [int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)]
    np.testing.assert_allclose(box, expected, atol=1)
