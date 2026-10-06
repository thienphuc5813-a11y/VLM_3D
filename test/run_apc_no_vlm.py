'''
LOCAL_RUN_PLAN.md muc 4.4: chay toan bo APC TRU VLM tren anh demo.

- object.__new__(APC) de bo qua __init__ (khong nap Qwen). Day la meo cho test, khong phai API chinh thuc.
- Danh sach vat, ref viewer, cau hoi duoc hardcode thay cho VLM (CASES ben duoi).
- Dung test/configs/no_vlm.yaml (use_vlm_refinement: false) => di qua nhanh code da sua o B2.
  GroundingDINO chay CPU (may nay khong co _C); B5 da sua nen device="cpu" duoc truyen dung vao predict().
- Phan render copy tu do_perspective_prompting_visual (doan truoc khi goi VLM), chay tren deepcopy.
- Duong ne duy nhat con lai la do MOI TRUONG (khong phai bug APC): keep_scene_viz_on_cpu().

Chay trong WSL tu goc repo:
    python test/run_apc_no_vlm.py                 # ca 4 anh demo
    python test/run_apc_no_vlm.py --case man      # 1 anh
Dau ra: test/outputs/<case>/
'''
import argparse
import copy
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml
from box import Box
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)   # duong dan checkpoint tuong doi; vision_tower.py:107 dung cache_dir='./'
sys.path.insert(0, REPO_ROOT)
for p in ["apc/vision_modules", "apc/vision_modules/src/omni3d",
          "apc/vision_modules/src/orient_anything", "apc/vision_modules/src/GroundingDINO"]:
    sys.path.append(p)

import apc.vision_modules.detection as det_mod                      # noqa: E402
from apc.apc_pipeline import APC                                    # noqa: E402
from apc.prompts import PromptParser                                # noqa: E402
from apc.renderer import RenderModule                               # noqa: E402
from apc.vision_modules import DepthModule, OrientationModule      # noqa: E402
from segment_anything import SamPredictor                           # noqa: E402

CASES = {
    "man": {
        "image": "demo/sample_image_man.jpg",
        "objects": ["person", "table"],
        "ref": "person",
        "question": "If I stand at the person's position facing where it is facing, "
                    "is the table on the left or on the right of me?",
    },
    "dog_chair": {
        "image": "demo/sample_dog_chair.jpg",
        "objects": ["dog", "cat"],
        "ref": "dog",
        "question": "From the dog's perspective, is the cat on the left or on the right?",
    },
    "spiderman": {
        "image": "demo/sample_image_spiderman.png",
        "objects": ["red spiderman", "yellow spiderman", "green spiderman"],
        "ref": "red spiderman",
        "question": "From the red spiderman's perspective, is the yellow spiderman on the left or on the right?",
    },
    "woman_chair": {
        "image": "demo/sample_woman_chair.png",
        "objects": ["woman", "chair", "penguin"],
        "ref": "woman",
        "question": "From the woman's perspective, is the chair on the left or on the right?",
    },
}

# copy y nguyen apc_pipeline.py:84-93
COLOR_DICT = [
    ['blue', [0, 0, 1]], ['red', [1, 0, 0]], ['green', [0, 1, 0]], ['yellow', [1, 1, 0]],
    ['purple', [1, 0, 1]], ['orange', [1, 0.5, 0]], ['brown', [0.5, 0.25, 0]], ['gray', [0.5, 0.5, 0.5]],
]


def keep_scene_viz_on_cpu():
    '''cubercnn/vis/vis.py:262 hardcode .cuda(); pytorch3d o day build CPU-only nen rasterize tren GPU loi.'''
    import apc.vision_modules.depth as depth_mod
    g = depth_mod.cube_vis.draw_scene_view.__globals__
    orig_join = g["join_meshes_as_scene"]

    def join_on_cpu(*args, **kwargs):
        meshes = orig_join(*args, **kwargs)
        meshes.cuda = lambda *a, **k: meshes
        return meshes

    g["join_meshes_as_scene"] = join_on_cpu


def build_apc(config, gpu="cuda"):
    keep_scene_viz_on_cpu()
    apc = object.__new__(APC)
    apc.config = config
    apc.logger = logging.getLogger("apc_no_vlm")
    apc.device_vlm = None
    apc.device_vision = gpu
    apc.vlm_model = None

    det = det_mod.DetectionModule(config, "cpu")          # GDINO o CPU (khong co _C)
    det.segmentation_model.to(gpu)                        # SAM o GPU
    det.segmentation_predictor = SamPredictor(det.segmentation_model)

    apc.detection_module = det
    apc.depth_module = DepthModule(config, gpu)
    apc.orientation_module = OrientationModule(config, gpu)
    apc.render_module = RenderModule(device="cpu")
    apc.prompt_parser = PromptParser(config)
    apc.color_dict = COLOR_DICT
    return apc


def to_jsonable(scene):
    return {k: {kk: np.asarray(vv).tolist() if kk != "color" else vv for kk, vv in v.items()}
            for k, v in scene.items()}


def render_visual_prompt(apc, abstract_scene_dict_ref, ref_viewer, apc_args, question):
    '''Copy apc_pipeline.py:419-447 (truoc buoc goi VLM), chay tren ban sao de khong dinh B4.'''
    abstract_scene_dict_ref = copy.deepcopy(abstract_scene_dict_ref)
    scene_ = {}
    for obj_idx, (obj_name, obj_dict) in enumerate(abstract_scene_dict_ref.items()):
        position = obj_dict['position']
        orientation = obj_dict['orientation']
        position[1:] = -position[1:]
        orientation[1:] = -orientation[1:]
        if obj_name == ref_viewer:
            position = np.array([0, 0, 0])
            orientation = np.array([0, 0, 1])
        scene_[obj_name] = {'position': position, 'orientation': orientation, 'color': apc.color_dict[obj_idx]}

    renderer_input = copy.deepcopy(scene_)
    visual_prompt = apc.render_module.render_visual_prompt(scene_, ref_viewer, **apc_args).resize((512, 512))
    prompt_abstract, color_obj_map = apc.prompt_real_to_abstract(question, renderer_input, ref_viewer)
    return visual_prompt, renderer_input, prompt_abstract, color_obj_map


def run_case(apc, name, case):
    out_dir = os.path.join(REPO_ROOT, "test", "outputs", name)
    os.makedirs(out_dir, exist_ok=True)
    image = Image.open(case["image"]).convert("RGB")
    apc_args = {
        "trace_save_dir": out_dir,
        "visualize_trace": True,                 # can True de co box3D mesh cho scene_abstraction.png
        "visualize_scene_abstraction": True,
        "render_whole_scene": False,
        "options": None,
    }
    log = {"case": name, **case}
    t0 = time.time()

    scene_cam = apc.do_scene_abstraction(image, case["objects"], **apc_args)
    log["detected"] = [k for k in scene_cam if k != "camera"]
    log["missing"] = [o for o in case["objects"] if o not in scene_cam]
    with open(os.path.join(out_dir, "abstract_camera.json"), "w") as f:
        json.dump(to_jsonable(scene_cam), f, indent=2)

    ref = case["ref"]
    if ref not in scene_cam:
        log["error"] = f"ref viewer '{ref}' khong duoc phat hien -> APC goc se loi KeyError o do_perspective_change"
    else:
        scene_ref = apc.do_perspective_change(copy.deepcopy(scene_cam), ref_viewer=ref)
        with open(os.path.join(out_dir, "abstract_ref.json"), "w") as f:
            json.dump(to_jsonable(scene_ref), f, indent=2)

        visual_prompt, renderer_input, prompt_abstract, color_obj_map = render_visual_prompt(
            apc, scene_ref, ref, apc_args, case["question"])
        visual_prompt.save(os.path.join(out_dir, "visual_prompt.png"))
        with open(os.path.join(out_dir, "renderer_input_opengl.json"), "w") as f:
            json.dump(to_jsonable(renderer_input), f, indent=2)
        log["prompt_abstract"] = prompt_abstract
        log["color_obj_map"] = color_obj_map
        # Goi y doc nhanh (he OpenCV cua viewer: x>0 la phai, z>0 la truoc mat)
        log["viewer_frame_summary"] = {
            k: {"side": "phai" if v["position"][0] > 0 else "trai",
                "front_or_back": "truoc" if v["position"][2] > 0 else "sau (bi loai khoi render)",
                "position": np.round(v["position"], 3).tolist()}
            for k, v in scene_ref.items() if k != ref
        }

    log["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "run_log.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(json.dumps(log, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=list(CASES), action="append")
    parser.add_argument("--config", default="test/configs/no_vlm.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = Box(yaml.safe_load(f))
    if config.detection.use_vlm_refinement:
        sys.exit("Khong co VLM: config phai dat detection.use_vlm_refinement: false")

    apc = build_apc(config)
    for name in args.case or list(CASES):
        print(f"\n######## {name} ########")
        try:
            run_case(apc, name, CASES[name])
        except Exception as e:
            logging.exception(f"case {name} loi: {e}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
