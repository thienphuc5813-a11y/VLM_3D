# Kế hoạch chạy và sửa APC trên máy local (step by step)

> Dành cho một Claude session khác thực hiện. Đọc theo thứ tự:
> `APC_project_context.md` → `test/TEST_PLAN_CONTEXT.md` → file này.
> Khi file này mâu thuẫn với `TEST_PLAN_CONTEXT.md`, **file này đúng hơn** (xem mục 1).
> Nhãn nguồn: `[CODE]` (file:dòng), `[ĐÃ KIỂM TRA]` (đã chạy lệnh trên máy này),
> `[SUY ĐOÁN]`, `[CHƯA XÁC MINH]`.

---

## 0. Sự thật về máy này `[ĐÃ KIỂM TRA ngày 2026-10-04]`

| Mục | Trạng thái |
|---|---|
| GPU | RTX 3050 6GB **Laptop**, driver 566.03 |
| WSL2 | Có `Ubuntu-22.04` (WSL version 2), đang tắt |
| Python / conda trên Windows | **Không có** trong PATH |
| `apc/vision_modules/src/` (GroundingDINO, ml-depth-pro, orient_anything, omni3d) | **Chưa có** |
| `apc/vision_modules/src/checkpoints/` | **Chưa có**. 4 checkpoint mới chỉ có trên Kaggle |
| Đường dẫn repo | Có ký tự non-ASCII (`kì 4`) và dấu cách, dễ làm hỏng build C++/CUDA |
| Git | Repo **chưa phải git repo** |

**Quyết định môi trường:** chạy mọi thứ trong **WSL2 Ubuntu-22.04**, với repo copy vào
filesystem của Linux (`~/apc-vlm`). Không build trên Windows gốc, không chạy từ `/mnt/c/...`,
vì chậm và đường dẫn có `kì 4`.

---

## 1. Đính chính `TEST_PLAN_CONTEXT.md` (đã đối chiếu với code)

1. **Dòng 34-36 sai.** `[x_3D, -y_3D, -z]` ([depth.py:235]) **không** cho ra "x phải, y lên,
   z tiến". Sau đó `flip_matrix` ([depth.py:248-249]) lật y và z lại, nên vị trí cuối cùng theo
   **OpenCV: x phải, y xuống, z tiến**. Chỉ khi render mới đổi sang OpenGL (y lên, z lùi) ở
   [apc_pipeline.py:424-425].
2. **Dòng 38-39 sai.** `transform_src_to_tgt` dùng **cả vector 3D** làm forward, gồm yaw và
   pitch (không có roll), với up toàn cục là `[0,-1,0]`. Biến `up_vector_cam` thực chất trỏ
   **xuống** (khớp OpenCV). Nếu forward song song với up thì kết quả là NaN.
3. **Dòng 43-47, 53-58 cần sửa.**
   - Cube có alpha 0.5, tức **bán trong suốt** ([renderer.py:181]).
   - Vật có z ≥ 0 (nằm sau viewer) bị **loại khỏi render** ([renderer.py:230-231]).
   - Sau khi chuẩn hóa, mọi vật phía trước đều nằm trong FOV, nên "visible" thực chất là
     "nằm ở nửa không gian phía trước".
   - Orientation có xoay cube, nhưng cube đối xứng và mọi mặt cùng màu nên **hướng không hề
     thấy được** trong ảnh render.
4. **Mục 3 (đặt `use_vlm_refinement: false`) sẽ dẫn vào bug B2.** Box sẽ ở dạng cxcywh chuẩn
   hóa thay vì xyxy pixel. Khi gọi module trực tiếp, code test **phải tự đổi box** giống
   [detection.py:118-126].
5. **Thiếu một bước quan trọng:** có thể chạy **toàn bộ APC trừ VLM** bằng cách
   `object.__new__(APC)` rồi tự gán từng module (xem bước 4.3). Cách này hữu ích hơn nhiều so
   với chỉ test từng module rời.

---

## 2. Danh sách bug đã biết (mục tiêu sửa ở bước 6)

| ID | Vị trí `[CODE]` | Lỗi | Cách sửa đề xuất |
|---|---|---|---|
| B1 | renderer.py:230-231, 268-271 | Lọc `positions` (z<0) nhưng không lọc `orientations`/`colors`, nên khi có vật nằm sau viewer, cube bị **tô màu của vật khác**. `positions_ori_idx` được tính ra nhưng không dùng | Lọc cả 3 list theo cùng một tập chỉ số |
| B2 | apc_pipeline.py:300 | `use_vlm_refinement=false` thì `box2D = boxes[0]` vẫn là tensor cxcywh chuẩn hóa | Đổi sang xyxy pixel như detection.py:118-126 |
| B3 | apc_pipeline.py:149 với 576 | Hàm trả về `None` nhưng nơi gọi unpack thành 2 biến, gây TypeError | Trả về `(None, conv_history)` và xử lý ở `run_apc` |
| B4 | apc_pipeline.py:424-425; renderer.py:258-261 | Sửa tại chỗ mảng position trong scene dict. Gọi lần 2 (hoặc chạy Num sau Vis) sẽ nhận tọa độ đã bị sửa | `copy.deepcopy` trước khi lật dấu và scale |
| B5 | detection.py:88-94 | `predict()` của GroundingDINO mặc định `device="cuda"` và gọi `model.to(device)` `[CHƯA XÁC MINH theo bản GDINO]`, nên `device_vision=cuda:1` bị bỏ qua | Truyền `device=self.device` |
| B6 | renderer.py:217-225 | `render_whole_scene=True` mà `max_z<0` thì `z_trans` chưa được gán | Khởi tạo `z_trans` cho nhánh còn lại |
| B7 | apc_pipeline.py:84-93, 432 | Chỉ có 8 màu (tính cả ref và camera), nhiều vật hơn sẽ gây IndexError | Thêm màu hoặc báo lỗi rõ ràng |
| B8 | apc_pipeline.py:305 | DepthPro chạy lại cho **mỗi vật** (kết quả giống nhau) | Tính một lần trước vòng lặp |
| B9 | vlm_qwenvl2_5.py:66-71 | `max_new_tokens` trong config không được dùng, luôn là 1024 | Đọc từ config |

**Không "sửa" âm thầm** (đây là lựa chọn nghiên cứu, chỉ thêm option và để mặc định giữ nguyên
hành vi gốc):
- R1: tiêu cự cố định `f = 4·W` ([depth.py:185-186]), bỏ qua tiêu cự DepthPro ước lượng.
- R2: lọc ±10% quanh mode của depth ([depth.py:217-226]).
- R3: câu trả lời của VLM được thêm vào hội thoại với `role='system'`
  ([apc_pipeline.py:482-486, 504]).
- R4: parse index refinement bằng `str(i) in response` ([detection.py:171-175]).

---

## 3. Bước 1: Dựng môi trường WSL2

Mỗi bước có **tiêu chí xong**. Chưa đạt thì dừng lại, báo người dùng, không tự đoán tiếp.

**1.1 Mở WSL, kiểm tra GPU**
```bash
wsl -d Ubuntu-22.04
nvidia-smi                      # phải thấy RTX 3050
```
Xong khi: `nvidia-smi` trong WSL thấy GPU.

**1.2 Copy repo vào filesystem Linux**
```bash
mkdir -p ~/apc-vlm
cp -r "/mnt/c/Users/thien/code/kì 4/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051/." ~/apc-vlm/
cd ~/apc-vlm && git init && git add -A && git commit -m "baseline: upstream APC-VLM"
```
Commit baseline là **bắt buộc**, để mọi chỉnh sửa về sau đều xem được diff.
Có thể giữ bản trên Windows chỉ để đọc trong VSCode, hoặc mở trực tiếp `~/apc-vlm` bằng VSCode
Remote-WSL.

**1.3 Gói hệ thống**
```bash
sudo apt update && sudo apt install -y build-essential git wget ninja-build \
  libgl1 libglu1-mesa libegl1 xvfb fonts-dejavu-core
```

**1.4 CUDA toolkit 12.4 trong WSL** (cần `nvcc` để build pytorch3d, detectron2 và custom op
của GroundingDINO). Cài theo hướng dẫn "WSL-Ubuntu" của NVIDIA, **chỉ toolkit, không cài
driver** trong WSL.
```bash
export CUDA_HOME=/usr/local/cuda-12.4; export PATH=$CUDA_HOME/bin:$PATH
nvcc --version                  # phải là 12.4
```

**1.5 Miniconda + env**
```bash
conda create -n apc_vlm python=3.10 -y && conda activate apc_vlm
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch;print(torch.cuda.is_available(), torch.version.cuda)"   # True 12.4
```

**1.6 Clone vision modules bằng HTTPS** (`setup/setup_vision_modules.sh` dùng `git@`, sẽ fail
nếu không có SSH key). Làm tay theo đúng cấu trúc script:
```bash
cd ~/apc-vlm/apc/vision_modules/src   # mkdir -p nếu chưa có
git clone https://github.com/IDEA-Research/GroundingDINO.git
git clone https://github.com/apple/ml-depth-pro.git
git clone https://github.com/SpatialVision/Orient-Anything.git orient_anything
git clone https://github.com/facebookresearch/omni3d.git
```
Ghi lại commit hash của từng repo vào `test/ENV_VERSIONS.md`, vì upstream có thể thay đổi.

**1.7 Cài các package phải build** (máy laptop RAM ít nên đặt `MAX_JOBS=2` để tránh OOM khi
build):
```bash
export MAX_JOBS=2
pip install numpy==1.23.4
cd apc/vision_modules/src/GroundingDINO && pip install -e . --no-build-isolation && cd -
cd apc/vision_modules/src/ml-depth-pro && pip install -e . && cd -
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install python-box trimesh==4.7.4 open3d==0.19.0 qwen-vl-utils accelerate \
  pyglet==1.5.27 opencv-python
pip install "transformers==4.49.0"    # [CHƯA XÁC MINH] bản tối thiểu có Qwen2_5_VL; ghim lại bản chạy được
pip install numpy==1.23.4             # cài lại phòng khi bị nâng lên
```
Pin version giống môi trường Kaggle đã chạy được: pytorch3d 0.7.8, detectron2 0.6,
numpy 1.23.4 (`APC_project_context.md` mục 3).

Xong khi lệnh sau chạy không lỗi, và GroundingDINO **không** in cảnh báo
"Failed to load custom C++ ops":
```bash
python -c "import pytorch3d, detectron2, groundingdino, depth_pro, segment_anything, open3d, trimesh; \
from groundingdino.models.GroundingDINO.ms_deform_attn import _C; print('ok')"
```

**1.8 Checkpoint (~6 GB)**: chạy phần download trong `setup/setup_vision_modules.sh`
(dòng 40-57) từ gốc repo, hoặc copy từ Kaggle. Xong khi có đủ:
`apc/vision_modules/src/checkpoints/{groundingdino_swint_ogc.pth, sam_vit_h_4b8939.pth, depth_pro.pt}`
và thư mục cache `models--Viglong--Orient-Anything` trong cùng thư mục đó.

**1.9 Màn hình cho render**: trimesh `scene.save_image()` cần OpenGL qua pyglet.
```bash
echo $DISPLAY                    # WSLg thường đặt sẵn :0
python -c "import trimesh; s=trimesh.Scene([trimesh.creation.box()]); open('/tmp/t.png','wb').write(s.save_image(resolution=(256,256))); print('render ok')"
```
Nếu fail thì dùng `xvfb-run -a python ...` cho mọi script có render.
**Lưu ý:** `run_APC.py:17` ghi đè `DISPLAY=':1'`. Script test **không** được import
`run_APC.py`.

**1.10 Smoke import**: chạy từ gốc repo, tự thêm 3 sys.path giống [apc_pipeline.py:15-17]:
```bash
python -c "import sys; [sys.path.append(p) for p in ['apc/vision_modules/src/omni3d','apc/vision_modules/src/orient_anything','apc/vision_modules/src/GroundingDINO']]; \
import apc.vision_modules, apc.renderer; print('import ok')"
```
Import `apc.vision_modules` sẽ kéo theo `cubercnn` (detectron2, pytorch3d) và Orient-Anything.
Nếu lỗi thì ghi lại traceback, **không** sửa trong `apc/`.

---

## 4. Bước 2-4: Viết test trong `test/` (không sửa `apc/`)

Quy ước chung:
- Dùng `pytest`. Chạy từ gốc repo. `test/conftest.py` lo phần `sys.path` (mục 1.10).
- Bug chưa sửa thì viết test theo **hành vi đúng** và đánh dấu
  `@pytest.mark.xfail(strict=True, reason="B1")`. Sửa xong ở bước 6, test tự chuyển sang pass
  (strict sẽ báo nếu bug biến mất mà quên bỏ xfail).
- Output ảnh ghi vào `test/outputs/` (thêm vào `.gitignore`).
- `object.__new__(Class)` là mẹo để **bỏ qua `__init__` nạp model**. Ghi chú rõ trong code
  rằng đây không phải API chính thức.

### 4.1 `test/test_geometry.py` (CPU, không cần checkpoint)

Giá trị kỳ vọng đã tính tay. Quy ước OpenCV: x phải, y xuống, z tiến.

`transform_src_to_tgt` ([vision_utils.py:76-128]):
| Viewer forward | Gốc viewer | Điểm (hệ camera) | Kỳ vọng (hệ viewer) | Ý nghĩa |
|---|---|---|---|---|
| `[0,0,1]` | `[0,0,0]` | `[1,2,3]` | `[1,2,3]` | đồng nhất |
| `[0,0,-1]` | `[0,0,0]` | `[1,0,5]` | `[-1,0,-5]` | viewer quay mặt về camera: phải và trái đảo nhau |
| `[1,0,0]` | `[0,0,0]` | `[0,0,5]` | `[-5,0,0]` | viewer nhìn sang phải camera: điểm xa camera nằm bên **trái** viewer |
| `[0,0,1]` | `[2,0,0]` | `[3,0,1]` | `[1,0,1]` | tịnh tiến |
| `[0,-1,0]` | bất kỳ | bất kỳ | NaN | ghi nhận suy biến (test tài liệu hóa, không phải bug cần sửa ngay) |

Thêm một test cho orientation: viewer `[0,0,-1]`, vật có hướng `[0,0,-1]` thì hướng trong hệ
viewer là `[0,0,1]`.

`OrientationModule.orientation_to_direction` ([orientation.py:56-65]), dùng
`object.__new__(OrientationModule)`:
- az=0, pol=0 cho `[0,0,-1]` (vật nhìn về camera).
- az=90, pol=0 cho `[-1,0,0]`. Quy ước azimuth của Orient-Anything `[CHƯA XÁC MINH]`: chỉ test
  công thức, ghi chú phần này.

`DepthModule.unproject_to_3D` ([depth.py:169-280]), dùng `object.__new__(DepthModule)`:
- Ảnh 200×100 (W×H), depth hằng `d=2.0` (float32), mask là hình chữ nhật đặt tâm tại pixel
  (u,v) = (150, 30). Kỳ vọng `pos3D ≈ (d·(u−100)/(4·200), d·(v−50)/(4·200), d)
  = (0.125, −0.05, 2.0)`, sai số 1e-2. Đây là test cho **quy ước trục và tiêu cự 4W**.
- Depth hai mức: 60% mask ở 2.0, 40% ở 3.0. Kỳ vọng z ≈ 2.0, chứng tỏ bộ lọc mode ±10% đã
  loại mức 3.0.

`APC.do_perspective_change` và `APC.prompt_real_to_abstract`: dùng `object.__new__(APC)`, gán
`apc.logger = logging.getLogger("test")`. Cả hai là hàm thuần.
- Kiểm tra key của ref viewer có vị trí `[0,0,0]` và hướng `[0,0,1]`, đồng thời `camera`
  cũng nằm trong output.
- `prompt_real_to_abstract`: "is the chair left of the dog" với map {chair: red, dog: green}.
  Thêm một ca **tên lồng nhau** ("person" và "person wearing a hat") để ghi nhận hành vi.

### 4.2 `test/test_renderer.py` (CPU, cần OpenGL ở bước 1.9)

- `RenderModule(device="cpu")`. Lưu ý `make_cube` gọi `look_at_view_transform(device=self.device)`.
- Đầu vào là dict **đã ở hệ OpenGL** (z<0 là phía trước), giống những gì
  `do_perspective_prompting_visual` truyền vào. Mỗi vật có `color = [tên, [r,g,b]]`.
- **Test B1 (xfail):** ref, A ở `[0,0,+5]` (sau lưng, màu đỏ), B ở `[1,0,-5]` (trước mặt, màu
  xanh lá). Monkeypatch `RenderModule.make_cube` để ghi lại `color` của từng lần gọi rồi gọi
  hàm gốc. Kỳ vọng đúng: chỉ có 1 cube và nó màu **xanh lá**. Code hiện tại sẽ tô **đỏ**.
- **Test B4 (xfail):** sau khi render, `position` trong dict đầu vào phải giữ nguyên.
- **Test B6 (xfail):** `render_whole_scene=True` với mọi vật có z<0 thì không được crash.
- Kiểm tra bằng mắt: lưu `test/outputs/render_*.png` cho 2-3 cấu hình (vật bên trái/phải,
  gần/xa), rồi xác nhận cube nằm đúng phía.

### 4.3 `test/test_vision_modules.py` (GPU, từng module riêng, đo VRAM)

Chạy **tuần tự, mỗi module trong một process riêng** (`pytest -k`), vì 6GB rất sát
`[SUY ĐOÁN]`: SAM ViT-H khoảng 2.5 GB weight fp32, Orient-Anything khoảng 1.2 GB, GDINO khoảng
0.7 GB, DepthPro fp16 khoảng 1 GB weight cộng activation lớn ở 1536px.

Config: copy `apc/configs/qwenvl2_5_7b_instruct.yaml` thành
`test/configs/no_vlm.yaml`, chỉ đổi `use_vlm_refinement: false`.

Thứ tự và tiêu chí (ghi `torch.cuda.max_memory_allocated()` vào `test/outputs/vram.md`):
1. `OrientationModule`: crop người trong `demo/sample_image_man.jpg`, in az/pol/rot và vector.
2. `DetectionModule`:
   - `run_detection(image_processed, "person")` trả về tensor cxcywh chuẩn hóa.
   - **Tự đổi** sang xyxy pixel (B2) rồi mới gọi `run_segmentation`. Lấy `masks[2]`.
   - Lưu overlay mask.
   - Nếu GPU 6GB không đủ cho ảnh full-res (GDINO đang tắt resize ở detection.py:68), ghi
     lại và thử với ảnh nhỏ hơn.
3. `DepthModule.run_depth_estimation`: lưu depth dạng ảnh màu, in min/median/max. Người trong
   ảnh demo phải có depth hợp lý (vài mét).
4. Thử nạp cả 3 module cùng lúc. Nếu OOM thì **ghi nhận, không phải lỗi**: dùng chiến lược
   nạp, chạy, `del`, `torch.cuda.empty_cache()` cho bước 4.4.

### 4.4 `test/run_apc_no_vlm.py`: toàn bộ APC trừ VLM (bước có giá trị nhất)

```python
apc = object.__new__(APC)              # bỏ qua __init__ (không nạp Qwen)
apc.config, apc.logger = config, logging.getLogger("apc")
apc.detection_module = DetectionModule(config, "cuda")
apc.depth_module = DepthModule(config, "cuda")
apc.orientation_module = OrientationModule(config, "cuda")
apc.render_module = RenderModule("cuda")
apc.prompt_parser = PromptParser(config)
apc.color_dict = [...]                 # copy y nguyên apc_pipeline.py:84-93
apc.vlm_model = None
```
- Né B2 mà **không sửa `apc/`**: giữ `use_vlm_refinement: true` và gán
  `apc.detection_module.run_detection_refinement = <hàm trả về box[0] đã đổi sang xyxy pixel>`.
- Danh sách vật và ref viewer **hardcode** thay cho VLM. Ví dụ với
  `demo/sample_image_man.jpg`: `["person", "table"]`, ref = `"person"`, câu hỏi lấy từ
  README.
- Gọi lần lượt `do_scene_abstraction`, `do_perspective_change`, rồi phần render trong
  `do_perspective_prompting_visual` (chỉ đoạn [apc_pipeline.py:419-447], vì đoạn sau gọi VLM).
  Có thể copy đoạn này vào script test.
- Nếu OOM thì nạp và giải phóng module tuần tự trong script này.
- Đầu ra: `scene_abstraction.png`, `visual_prompt.png`, file JSON tọa độ trong hệ camera và hệ
  viewer, tất cả trong `test/outputs/<tên ảnh>/`.
- Xong khi: chạy được cả 4 ảnh demo, và **người dùng xác nhận bằng mắt** rằng vị trí trái/phải
  trong `visual_prompt.png` khớp trực giác.

### 4.5 (Tùy chọn) Smoke test có VLM trên 6GB

- Qwen2.5-VL-**7B** gần như chắc chắn không vừa 6GB, kể cả ở 4-bit, khi còn phải chứa vision
  module `[SUY ĐOÁN]`.
- Chỉ để test đường ống end-to-end: dùng `Qwen/Qwen2.5-VL-3B-Instruct` 4-bit (bitsandbytes),
  qua một subclass của `ModelQwenVL2_5` đặt trong `test/` (override `load_model`), gán vào
  `apc.vlm_model`.
- **Số liệu từ bước này không so được với paper.** Ghi rõ điều đó trong mọi output.

---

## 5. Bước 5: Điểm dừng, hỏi người dùng

Báo cáo cho người dùng: kết quả `pytest -v` (bao nhiêu pass, bao nhiêu xfail), bảng VRAM, và
ảnh từ 4.4. **Chờ người dùng đồng ý** rồi mới sang bước 6, vì bước 6 sửa code gốc trong `apc/`
(ngược với ràng buộc của `TEST_PLAN_CONTEXT.md`).

## 6. Bước 6: Sửa bug trong `apc/` (chỉ khi được đồng ý)

- Mỗi bug B1 đến B9 là **một commit riêng** (`fix(B1): ...`). Sau mỗi commit chạy lại
  `pytest`, và test xfail tương ứng phải chuyển sang pass.
- R1 đến R4: chỉ thêm option vào config, **mặc định giữ hành vi gốc**, ví dụ
  `depth.use_estimated_focal: false`.
- Không đổi prompt template trong `apc/prompts.py`.
- Cuối bước: chạy lại 4.4 trên 4 ảnh demo, so ảnh trước và sau fix (B1 làm đổi màu cube khi có
  vật sau lưng viewer).

## 7. Ngoài phạm vi local (để dành cho Kaggle)

- Chạy Qwen2.5-VL-7B đầy đủ.
- Viết APC-Num: hiện chưa có, [apc_pipeline.py:617-619]. Tọa độ hệ viewer lấy từ
  `abstract_scene_dict[ref_viewer]` theo OpenCV, nên phải lật y nếu prompt dùng quy ước
  "y lên". Không lật z.
- Viết harness đánh giá trên 3DSRBench/COMFORT++ (repo không có), oracle abstraction, router
  Num/Vis.

## 8. Checklist file đầu ra trong `test/`

```
test/
  TEST_PLAN_CONTEXT.md, LOCAL_RUN_PLAN.md
  ENV_VERSIONS.md            # pip freeze + commit hash các repo con (bước 1.6-1.7)
  conftest.py                # sys.path
  configs/no_vlm.yaml
  test_geometry.py           # 4.1
  test_renderer.py           # 4.2
  test_vision_modules.py     # 4.3
  run_apc_no_vlm.py          # 4.4
  outputs/                   # gitignored: ảnh, vram.md, json
```
