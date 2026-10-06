'''
LOCAL_RUN_PLAN.md muc 4.2: RenderModule (apc/renderer.py), CPU, can OpenGL (WSLg DISPLAY=:0).

Dau vao la dict DA O HE OpenGL (z<0 la phia truoc), giong nhung gi
do_perspective_prompting_visual truyen vao (apc_pipeline.py:419-439).
Bug chua sua duoc viet theo hanh vi DUNG va danh dau xfail(strict=True).

Chay:  pytest test/test_renderer.py -v
'''
import copy
import os

import numpy as np
import pytest

renderer_mod = pytest.importorskip("apc.renderer", reason="apc.renderer can pytorch3d + trimesh")
RenderModule = renderer_mod.RenderModule

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _obj(pos, color_name, rgb):
    return {
        "position": np.array(pos, dtype=float),
        "orientation": np.array([0.0, 0.0, 1.0]),
        "color": [color_name, rgb],
    }


def _ref():
    return _obj([0, 0, 0], "blue", [0, 0, 1])


def _args(tmp_path, whole_scene=False):
    return {
        "trace_save_dir": str(tmp_path),
        "visualize_trace": False,
        "render_whole_scene": whole_scene,
        "options": None,
    }


@pytest.fixture
def render_module():
    return RenderModule(device="cpu")


@pytest.fixture
def recorded_colors(monkeypatch):
    calls = []
    original = RenderModule.make_cube

    def spy(self, pos, ori, box_size=2, color=None):
        calls.append(tuple(np.round(color[0][:3], 3)))
        return original(self, pos, ori, box_size=box_size, color=color)

    monkeypatch.setattr(RenderModule, "make_cube", spy)
    return calls


def _render_or_skip(render_module, scene, ref, args):
    try:
        return render_module.render_visual_prompt(scene, ref, **args)
    except Exception as e:
        # Chi skip khi loi do thieu OpenGL context; loi khac phai noi len
        msg = f"{type(e).__name__}: {e}"
        if any(k in msg.lower() for k in ("pyglet", "display", "gl", "context", "window")):
            pytest.skip(f"khong co OpenGL context: {msg}")
        raise


# --------------------------------------------------------------------- #
def test_B1_object_behind_viewer_does_not_shift_colors(render_module, recorded_colors, tmp_path):
    scene = {
        "person": _ref(),
        "A_behind": _obj([0, 0, 5], "red", [1, 0, 0]),      # sau lung => bi loai
        "B_front": _obj([1, 0, -5], "green", [0, 1, 0]),    # truoc mat
    }
    _render_or_skip(render_module, scene, "person", _args(tmp_path))
    assert recorded_colors == [(0.0, 1.0, 0.0)], f"mau cube duoc ve: {recorded_colors}"


def test_B4_input_positions_not_mutated(render_module, tmp_path):
    scene = {"person": _ref(), "B_front": _obj([1, 0, -5], "green", [0, 1, 0])}
    before = copy.deepcopy(scene)
    _render_or_skip(render_module, scene, "person", _args(tmp_path))
    np.testing.assert_allclose(scene["B_front"]["position"], before["B_front"]["position"])


def test_B6_render_whole_scene_all_in_front(render_module, tmp_path):
    scene = {
        "person": _ref(),
        "A": _obj([-1, 0, -4], "red", [1, 0, 0]),
        "B": _obj([1, 0, -6], "green", [0, 1, 0]),
    }
    _render_or_skip(render_module, scene, "person", _args(tmp_path, whole_scene=True))


# --------------------------------------------------------------------- #
def _color_mass(img):
    arr = np.array(img.convert("RGB")).astype(np.int16)
    half = arr.shape[1] // 2
    red = (arr[:, :, 0] - (arr[:, :, 1] + arr[:, :, 2]) // 2) > 20
    green = (arr[:, :, 1] - (arr[:, :, 0] + arr[:, :, 2]) // 2) > 20
    return {
        "red_left": int(red[:, :half].sum()), "red_right": int(red[:, half:].sum()),
        "green_left": int(green[:, :half].sum()), "green_right": int(green[:, half:].sum()),
    }


@pytest.mark.parametrize("name, red_x, green_x", [("red_left_green_right", -2, 2), ("red_right_green_left", 2, -2)])
def test_left_right_placement(render_module, tmp_path, name, red_x, green_x):
    scene = {
        "person": _ref(),
        "red_obj": _obj([red_x, 0, -5], "red", [1, 0, 0]),
        "green_obj": _obj([green_x, 0, -7], "green", [0, 1, 0]),
    }
    img = _render_or_skip(render_module, scene, "person", _args(tmp_path))
    os.makedirs(OUT_DIR, exist_ok=True)
    img.save(os.path.join(OUT_DIR, f"render_{name}.png"))

    m = _color_mass(img)
    red_on_left = m["red_left"] > m["red_right"]
    green_on_left = m["green_left"] > m["green_right"]
    assert red_on_left == (red_x < 0), m
    assert green_on_left == (green_x < 0), m
