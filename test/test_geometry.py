'''
LOCAL_RUN_PLAN.md muc 4.1: logic hinh hoc thuan, CPU, khong can checkpoint.
Quy uoc OpenCV cua scene abstraction: x phai, y xuong, z tien.

Chay tu goc repo:  pytest test/test_geometry.py -v

- transform_src_to_tgt: nap thang file vision_utils.py (numpy thuan), chay duoc ngay.
- Cac test con lai can import apc.vision_modules (kem pytorch3d/detectron2 qua cubercnn);
  neu chua cai xong thi tu dong SKIP, khong FAIL.
- object.__new__(Class) duoc dung de bo qua __init__ nap model. Day la meo cho test,
  khong phai API chinh thuc cua APC.
'''
import importlib.util
import logging
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_vision_utils():
    # Khong import qua apc.vision_modules vi __init__ cua package keo theo ca 3 module nang
    path = os.path.join(REPO_ROOT, "apc", "vision_modules", "vision_utils.py")
    spec = importlib.util.spec_from_file_location("_apc_vision_utils_isolated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


transform_src_to_tgt = _load_vision_utils().transform_src_to_tgt


def _import_or_skip(modname):
    return pytest.importorskip(modname, reason=f"{modname} chua import duoc (moi truong chua du)")


# ===================================================================== #
# transform_src_to_tgt  (apc/vision_modules/vision_utils.py:76-128)
# ===================================================================== #
@pytest.mark.parametrize(
    "forward, origin, point, expected",
    [
        ([0, 0, 1], [0, 0, 0], [1, 2, 3], [1, 2, 3]),       # dong nhat
        ([0, 0, -1], [0, 0, 0], [1, 0, 5], [-1, 0, -5]),    # viewer quay ve camera: trai/phai dao
        ([1, 0, 0], [0, 0, 0], [0, 0, 5], [-5, 0, 0]),      # viewer nhin sang phai camera
        ([0, 0, 1], [2, 0, 0], [3, 0, 1], [1, 0, 1]),       # tinh tien
        ([0, 0, -1], [0, 0, 5], [2, 0, 5], [-2, 0, 0]),     # ban ben phai camera -> ben trai nguoi doi dien camera
    ],
)
def test_transform_position(forward, origin, point, expected):
    pos, _ = transform_src_to_tgt(
        np.array(point, dtype=float), np.array([0.0, 0.0, 1.0]),
        np.array(origin, dtype=float), np.array(forward, dtype=float),
    )
    np.testing.assert_allclose(pos, expected, atol=1e-3)


def test_transform_orientation():
    _, ori = transform_src_to_tgt(
        np.zeros(3), np.array([0.0, 0.0, -1.0]),
        np.zeros(3), np.array([0.0, 0.0, -1.0]),
    )
    np.testing.assert_allclose(ori, [0, 0, 1], atol=1e-3)


@pytest.mark.parametrize(
    "forward",
    [[0, 0, 1], [1, 0, 0], [0, 0, -1], [0.3, 0.1, -0.9], [-0.5, 0.2, 0.84], [0.6, -0.4, 0.7]],
)
def test_transform_is_proper_rotation(forward):
    # det = -1 nghia la lat guong: trai/phai bi doi am tham ma khong co loi nao bao ra
    forward = np.array(forward, dtype=float)
    R = np.stack(
        [transform_src_to_tgt(e, e, np.zeros(3), forward)[0] for e in np.eye(3)], axis=1
    )
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=2e-3)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=5e-3)


def test_transform_degenerate_vertical_viewer_gives_nan():
    # Tai lieu hoa hanh vi hien tai (khong phai bug can sua ngay):
    # forward // up_global [0,-1,0] => cross = 0 => chia 0 => NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        pos, _ = transform_src_to_tgt(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
            np.zeros(3), np.array([0.0, -1.0, 0.0]),
        )
    assert np.isnan(pos).any()


# ===================================================================== #
# OrientationModule.orientation_to_direction  (orientation.py:56-65)
# ===================================================================== #
@pytest.fixture(scope="module")
def orientation_module():
    mod = _import_or_skip("apc.vision_modules.orientation")
    return object.__new__(mod.OrientationModule)


@pytest.mark.parametrize(
    "azimuth, polar, expected",
    [
        (0, 0, [0, 0, -1]),     # vat nhin ve phia camera
        (90, 0, [-1, 0, 0]),    # quy uoc azimuth cua Orient-Anything [CHUA XAC MINH]: chi test cong thuc
    ],
)
def test_orientation_to_direction(orientation_module, azimuth, polar, expected):
    v = orientation_module.orientation_to_direction(azimuth, polar)
    np.testing.assert_allclose(v, expected, atol=1e-6)


# ===================================================================== #
# DepthModule.unproject_to_3D  (depth.py:169-280)
# ===================================================================== #
@pytest.fixture(scope="module")
def depth_module():
    from box import Box
    mod = _import_or_skip("apc.vision_modules.depth")
    dm = object.__new__(mod.DepthModule)
    dm.config = Box({"depth": {}})     # option R1/R2 khong co => hanh vi goc
    return dm


W, H = 200, 100


def _gray_image():
    from PIL import Image
    return Image.fromarray(np.full((H, W, 3), 128, dtype=np.uint8))


def _rect_mask(u_c, v_c, half=10):
    mask = np.zeros((H, W), dtype=np.float32)
    mask[v_c - half: v_c + half + 1, u_c - half: u_c + half + 1] = 1.0
    return mask


def test_unproject_constant_depth_axes_and_focal(depth_module):
    # Kiem tra quy uoc truc (OpenCV) va tieu cu co dinh f = 4W
    d, u, v = 2.0, 150, 30
    depth = np.full((H, W), 10.0, dtype=np.float32)
    mask = _rect_mask(u, v)
    depth[mask > 0.5] = d

    pos, _ = depth_module.unproject_to_3D(
        image=_gray_image(), depth=depth, segment_mask=mask, return_box3D=False
    )
    f = 4.0 * W
    expected = [d * (u - W / 2) / f, d * (v - H / 2) / f, d]   # = (0.125, -0.05, 2.0)
    np.testing.assert_allclose(pos, expected, atol=1e-2)


def test_unproject_mode_filter_drops_far_level(depth_module):
    # 60% mask o 2.0, 40% o 3.0 => mode = 2.0, loc +-10% loai muc 3.0
    depth = np.full((H, W), 10.0, dtype=np.float32)
    mask = _rect_mask(100, 50, half=20)
    ys, xs = np.where(mask > 0.5)
    cut = xs.min() + int(0.6 * (xs.max() - xs.min() + 1))
    depth[ys, xs] = np.where(xs < cut, 2.0, 3.0)

    pos, _ = depth_module.unproject_to_3D(
        image=_gray_image(), depth=depth, segment_mask=mask, return_box3D=False
    )
    assert pos[2] == pytest.approx(2.0, abs=1e-2)


def test_R2_mode_filter_biases_toward_camera_on_continuous_depth(depth_module):
    # Dac ta hanh vi hien tai cua R2 (lua chon nghien cuu, khong phai bug):
    # depth lien tuc nhu DepthPro thi moi gia tri gan nhu khac nhau, scipy mode tra ve gia
    # tri NHO NHAT, va bo loc +-10% chi giu dai gan camera nhat => tam bi keo ve phia camera.
    depth = np.full((H, W), 10.0, dtype=np.float32)
    mask = _rect_mask(100, 50, half=20)
    ys, xs = np.where(mask > 0.5)
    depth[ys, xs] = 2.0 + (xs - xs.min()) / (xs.max() - xs.min())   # doc tu 2.0 -> 3.0

    pos, _ = depth_module.unproject_to_3D(
        image=_gray_image(), depth=depth, segment_mask=mask, return_box3D=False
    )
    surface_mean = float(depth[ys, xs].mean())                       # ~2.5
    assert pos[2] < surface_mean - 0.25, f"z_est={pos[2]:.3f}, mean be mat={surface_mean:.3f}"


# ===================================================================== #
# APC.do_perspective_change / APC.prompt_real_to_abstract  (apc_pipeline.py)
# ===================================================================== #
@pytest.fixture(scope="module")
def apc():
    mod = _import_or_skip("apc.apc_pipeline")
    obj = object.__new__(mod.APC)
    obj.logger = logging.getLogger("test")
    return obj


def _scene_camera_frame():
    return {
        "camera": {"position": np.array([0.0, 0.0, 0.0]), "orientation": np.array([0.0, 0.0, 1.0])},
        "person": {"position": np.array([0.0, 0.0, 5.0]), "orientation": np.array([0.0, 0.0, -1.0])},
        "table": {"position": np.array([2.0, 0.0, 5.0]), "orientation": np.array([0.0, 0.0, 1.0])},
    }


def test_perspective_change_ref_viewer_at_origin(apc):
    out = apc.do_perspective_change(_scene_camera_frame(), ref_viewer="person")
    np.testing.assert_allclose(out["person"]["position"], [0, 0, 0])
    np.testing.assert_allclose(out["person"]["orientation"], [0, 0, 1])
    assert "camera" in out


def test_perspective_change_semantics(apc):
    out = apc.do_perspective_change(_scene_camera_frame(), ref_viewer="person")
    np.testing.assert_allclose(out["camera"]["position"], [0, 0, 5], atol=1e-3)     # camera truoc mat nguoi
    np.testing.assert_allclose(out["camera"]["orientation"], [0, 0, -1], atol=1e-3)  # camera nhin ve nguoi
    np.testing.assert_allclose(out["table"]["position"], [-2, 0, 0], atol=1e-3)     # ban ben trai nguoi


def _scene_with_colors(names_colors, ref="person"):
    scene = {
        "camera": {"position": np.zeros(3), "orientation": np.array([0, 0, 1]), "color": ["gray", [0.5] * 3]},
        ref: {"position": np.zeros(3), "orientation": np.array([0, 0, 1]), "color": ["blue", [0, 0, 1]]},
    }
    for name, color in names_colors:
        scene[name] = {"position": np.zeros(3), "orientation": np.array([0, 0, 1]), "color": [color, [0, 0, 0]]}
    return scene


def test_prompt_real_to_abstract_basic(apc):
    scene = _scene_with_colors([("chair", "red"), ("dog", "green")])
    prompt, cmap = apc.prompt_real_to_abstract("is the chair left of the dog", scene, "person")
    assert prompt == "is the red cube left of the green cube"
    assert "- red cube -> chair" in cmap
    assert "- green cube -> dog" in cmap


@pytest.mark.xfail(strict=True, reason="NEW-1: thay chuoi theo thu tu dict, ten ngan an mat ten long nhau")
def test_prompt_real_to_abstract_nested_names(apc):
    # 'person' dung truoc 'person wearing a hat' trong dict => ' person' bi thay truoc,
    # pha hong ten dai. Ky vong dung: moi vat co mau rieng.
    scene = _scene_with_colors([("person", "red"), ("person wearing a hat", "green")], ref="dog")
    prompt, _ = apc.prompt_real_to_abstract(
        "is the person wearing a hat left of the person", scene, "dog"
    )
    assert prompt == "is the green cube left of the red cube"
