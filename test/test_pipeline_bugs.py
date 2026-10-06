'''
LOCAL_RUN_PLAN.md muc 6: test cho cac bug chua co test o muc 4
(B2, B3, B4 phan apc_pipeline, B5, B7, B8, B9).

Dung stub thay cho model that nen chay tren CPU, khong can checkpoint, vai giay.
Bug chua sua => xfail(strict=True). Sua xong thi bo marker; strict bao loi neu quen bo.

Chay:  pytest test/test_pipeline_bugs.py -v
'''
import copy
import logging

import numpy as np
import pytest
import torch
from box import Box
from PIL import Image

apc_pipeline = pytest.importorskip("apc.apc_pipeline", reason="can moi truong day du")
APC = apc_pipeline.APC
from apc.prompts import PromptParser  # noqa: E402

W, H = 200, 100

# copy y nguyen apc_pipeline.py:84-93
COLOR_DICT = [
    ['blue', [0, 0, 1]], ['red', [1, 0, 0]], ['green', [0, 1, 0]], ['yellow', [1, 1, 0]],
    ['purple', [1, 0, 1]], ['orange', [1, 0.5, 0]], ['brown', [0.5, 0.25, 0]], ['gray', [0.5, 0.5, 0.5]],
]


def _config(refine=False):
    return Box({
        "vlm": {"max_new_tokens": 512, "do_sample": False, "temperature": 0.0},
        "detection": {"use_vlm_refinement": refine, "num_candidates": 5,
                      "box_threshold": 0.05, "text_threshold": 0.05},
        "depth": {},
    })


class StubDetection:
    def __init__(self):
        self.seg_boxes = []

    def detection_process_image(self, image):
        return None, torch.zeros(3, H, W)

    def run_detection(self, image_processed, category):
        return [torch.tensor([0.5, 0.5, 0.2, 0.4])]          # cxcywh chuan hoa, giong GroundingDINO

    def run_segmentation(self, image, box2d):
        self.seg_boxes.append(box2d)
        return np.ones((3, H, W))


class StubDepth:
    def __init__(self):
        self.calls = 0

    def run_depth_estimation(self, image):
        self.calls += 1
        return np.full((H, W), 2.0, dtype=np.float32)

    def unproject_to_3D(self, image, depth, segment_mask, return_box3D=False):
        return np.array([0.0, 0.0, 2.0]), None


class StubOrientation:
    def run_orientation_estimation(self, image, category=None, bbox=None, **apc_args):
        return np.array([0.0, 0.0, -1.0])


class StubRender:
    def render_visual_prompt(self, scene, ref_viewer, **apc_args):
        return Image.new("RGB", (512, 512), "white")


class StubVLM:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = []

    def process_messages(self, messages, **kwargs):
        self.calls.append([{"role": m["role"]} for m in messages])
        return self.replies.pop(0) if self.replies else "ok"


def _make_apc(config, vlm=None):
    apc = object.__new__(APC)          # bo qua __init__ (khong nap model); meo cho test
    apc.config = config
    apc.logger = logging.getLogger("test")
    apc.detection_module = StubDetection()
    apc.depth_module = StubDepth()
    apc.orientation_module = StubOrientation()
    apc.render_module = StubRender()
    apc.prompt_parser = PromptParser(config)
    apc.color_dict = COLOR_DICT
    apc.vlm_model = vlm or StubVLM()
    return apc


def _apc_args(tmp_path):
    return {"trace_save_dir": str(tmp_path), "visualize_trace": False,
            "visualize_scene_abstraction": False, "render_whole_scene": False, "options": None}


def _image():
    return Image.new("RGB", (W, H), "gray")


def _scene_ref(n_objects=2):
    scene = {
        "person": {"position": np.array([0.0, 0.0, 0.0]), "orientation": np.array([0.0, 0.0, 1.0])},
        "camera": {"position": np.array([0.0, 0.0, 5.0]), "orientation": np.array([0.0, 0.0, -1.0])},
    }
    for i in range(n_objects):
        scene[f"obj{i}"] = {"position": np.array([float(i) - 1.0, 0.5, 3.0]),
                            "orientation": np.array([0.0, 0.0, 1.0])}
    return scene


# --------------------------------------------------------------------- #
def test_B2_box_is_xyxy_pixel_without_refinement(tmp_path):
    apc = _make_apc(_config(refine=False))
    apc.do_scene_abstraction(_image(), ["chair"], **_apc_args(tmp_path))
    box = apc.detection_module.seg_boxes[0]
    assert isinstance(box, (list, tuple)), f"box truyen cho SAM la {type(box).__name__}: {box}"
    assert all(isinstance(v, int) for v in box)
    # cx.5 cy.5 w.2 h.4 tren anh 200x100; int() cat phan le nhu detection.py:118-126 (0.7f32 -> 69)
    np.testing.assert_allclose(box, [80, 30, 120, 70], atol=1)


def test_B3_run_apc_handles_unparseable_object_list(tmp_path):
    apc = _make_apc(_config(), vlm=StubVLM(["khong co dau ngoac", "van khong co"]))
    response, _ = apc.run_apc(_image(), "is the chair left of the dog?",
                              trace_save_dir=str(tmp_path), visualize_trace=False,
                              visualize_scene_abstraction=False)
    assert response is None


def test_B4_prompting_visual_does_not_mutate_input(tmp_path):
    apc = _make_apc(_config())
    scene = _scene_ref()
    before = copy.deepcopy(scene)
    apc.do_perspective_prompting_visual("person", scene, "is obj0 left of obj1?", **_apc_args(tmp_path))
    for name in scene:
        np.testing.assert_allclose(scene[name]["position"], before[name]["position"], err_msg=name)
        np.testing.assert_allclose(scene[name]["orientation"], before[name]["orientation"], err_msg=name)


def test_B5_detection_uses_module_device(monkeypatch):
    det_mod = pytest.importorskip("apc.vision_modules.detection")
    captured = {}

    def fake_predict(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.5, 0.5, 0.2, 0.4]]), torch.tensor([0.9]), ["x"]

    monkeypatch.setattr(det_mod, "predict", fake_predict)
    det = object.__new__(det_mod.DetectionModule)
    det.device = "cpu"
    det.config = _config()
    det.detection_model = object()
    det.run_detection(torch.zeros(3, H, W), "chair")
    assert captured.get("device") == "cpu"


def test_B7_too_many_objects_gives_clear_error(tmp_path):
    apc = _make_apc(_config())
    scene = _scene_ref(n_objects=7)          # 7 vat + ref + camera = 9 > 8 mau
    with pytest.raises(ValueError, match="8"):
        apc.do_perspective_prompting_visual("person", scene, "q", **_apc_args(tmp_path))


def test_B8_depth_estimated_once_per_image(tmp_path):
    apc = _make_apc(_config(refine=False))
    apc.detection_module.run_detection = lambda img, cat: [torch.tensor([0.5, 0.5, 0.2, 0.4])]
    apc.do_scene_abstraction(_image(), ["chair", "dog", "cat"], **_apc_args(tmp_path))
    assert apc.depth_module.calls == 1


# --------------------------------------------------------------------- #
class _Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.zeros(1, 3, dtype=torch.long))
        self.input_ids = self["input_ids"]

    def to(self, device):
        return self


class _StubProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "text"

    def __call__(self, **kwargs):
        return _Inputs()

    def batch_decode(self, ids, **kwargs):
        return ["ok"]


class _StubGenerator:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.zeros(1, 5, dtype=torch.long)


def test_B9_max_new_tokens_from_config():
    vlm_mod = pytest.importorskip("apc.vlms.vlm_qwenvl2_5")
    vlm = object.__new__(vlm_mod.ModelQwenVL2_5)   # bo qua __init__ (khong nap Qwen)
    vlm.config = Box({"max_new_tokens": 512})
    vlm.device = "cpu"
    vlm.processor = _StubProcessor()
    vlm.vlm_model = _StubGenerator()
    vlm.process_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert vlm.vlm_model.kwargs["max_new_tokens"] == 512
