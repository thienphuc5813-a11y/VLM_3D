# Phien ban moi truong

Sinh tu dong boi test/finish_step1.sh luc 2026-10-05T15:59:59+07:00.
WSL2 Ubuntu-22.04, venv `~/apc_env`, GPU RTX 3050 6GB Laptop, driver 566.03.

## Lech so voi LOCAL_RUN_PLAN.md

- Repo giu nguyen o `/mnt/c/...` (khong copy sang `~/apc-vlm`), venv o `~/apc_env` (khong dung conda).
- Khong co `nvcc` (pip wheel `nvidia-cuda-nvcc-cu12` chi co ptxas, khong co nvcc).
  GroundingDINO, detectron2, pytorch3d deu build CPU-only.
  GroundingDINO khong co `_C` => theo ms_deform_attn.py:330, DetectionModule phai chay o device=cpu.
- numpy 1.23.4 (theo setup/requirements.txt). scipy 1.15.3 bao can >=1.23.5 nhung import duoc.
  Trong apc/ chi co omni3d_evaluation.py dung np.float (APC khong goi), nen co the nang len 1.26.4 neu can.

## Commit cac repo con

```
GroundingDINO    856dde20aee659246248e20734ef9ba5214f5e44
ml-depth-pro     9e65e4dbe9568d23c546fcec53302b10445e109e
orient_anything  759282c26e924988c952d6d33212c48349dc9aff
omni3d           778bd0210e5e1b584cc52daeba57cae3a3ee8c4e
pytorch3d        V0.7.8 (75ebeeaea0908c5527e7b1e305fbc7681382db47)
```

## pip freeze

```
absl-py==2.5.0
accelerate==1.15.0
addict==2.4.0
annotated-types==0.8.0
antlr4-python3-runtime==4.9.3
asttokens==3.0.2
attrs==26.1.0
av==17.1.0
black==26.5.1
blinker==1.9.0
certifi==2026.7.22
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
comm==0.2.3
ConfigArgParse==1.8.0
contourpy==1.3.2
cycler==0.12.1
dash==4.4.1
decorator==5.3.1
defusedxml==0.7.1
depth_pro @ file:///mnt/c/Users/thien/code/k%C3%AC%204/AIL/github/KAIST-Visual-AI-Group-APC-VLM-d7ca051/apc/vision_modules/src/ml-depth-pro
detectron2 @ git+https://github.com/facebookresearch/detectron2.git@1e3e13bbf607b54f62205c4c33922521822fb298
exceptiongroup==1.3.1
executing==2.2.1
fastjsonschema==2.22.2
filelock==3.32.3
Flask==3.1.3
fonttools==4.65.0
fsspec==2026.7.0
fvcore==0.1.5.post20221221
groundingdino @ file:///home/tphuc58/build/GroundingDINO
grpcio==1.84.0
hf-xet==1.6.0
huggingface_hub==0.36.2
hydra-core==1.3.7
idna==3.20
importlib_metadata==9.0.1
iniconfig==2.3.0
iopath==0.1.9
ipython==8.39.0
ipywidgets==8.1.9
itsdangerous==2.2.0
janus==2.0.0
jedi==0.20.0
Jinja2==3.1.6
joblib==1.6.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter_core==5.9.1
jupyterlab_widgets==3.0.17
kiwisolver==1.5.1
Markdown==3.10.3
MarkupSafe==3.0.3
matplotlib==3.10.9
matplotlib-inline==0.2.2
mpmath==1.3.0
mypy_extensions==1.1.0
narwhals==2.26.0
nbformat==5.11.1
nest-asyncio==1.6.0
networkx==3.4.2
numpy==1.23.4
nvidia-cublas-cu12==12.4.2.65
nvidia-cuda-cccl-cu12==12.4.127
nvidia-cuda-cupti-cu12==12.4.99
nvidia-cuda-nvcc-cu12==12.4.131
nvidia-cuda-nvrtc-cu12==12.4.99
nvidia-cuda-runtime-cu12==12.4.99
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.0.44
nvidia-curand-cu12==10.3.5.119
nvidia-cusolver-cu12==11.6.0.99
nvidia-cusparse-cu12==12.3.0.142
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.4.99
nvidia-nvtx-cu12==12.4.99
omegaconf==2.3.1
open3d==0.19.0
opencv-python==5.0.0.93
packaging==26.3
pandas==2.3.3
parso==0.8.7
pathspec==1.1.1
pexpect==4.9.0
pillow==12.3.0
pillow_heif==1.8.0
platformdirs==4.12.3
plotly==7.1.0
pluggy==1.6.0
portalocker==4.4.0
prompt_toolkit==3.0.53
protobuf==7.36.2
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.4
pycocotools==2.0.11
pydantic==2.13.5
pydantic_core==2.46.5
pyDeprecate==0.11.0
pyglet==1.5.27
Pygments==2.21.0
pyparsing==3.3.3
pyquaternion==0.9.9
pytest==9.1.1
python-box==7.4.1
python-dateutil==2.9.0.post0
pytokens==0.4.1
pytorch3d @ file:///tmp/p3d_clone
pytz==2026.5
PyYAML==6.0.3
qwen-vl-utils==0.0.14
referencing==0.37.0
regex==2026.9.29
requests==2.34.2
retrying==1.4.2
rpds-py==0.30.0
safetensors==0.8.0
scikit-learn==1.7.2
scipy==1.15.3
segment_anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf
six==1.17.0
stack-data==0.6.3
supervision==0.30.7
sympy==1.14.0
tabulate==0.10.0
tensorboard==2.21.0
tensorboard-data-server==0.7.2
termcolor==3.3.0
threadpoolctl==3.7.0
timm==1.0.30
tokenizers==0.21.4
tomli==2.4.1
torch==2.4.1+cu124
torchvision==0.19.1+cu124
tqdm==4.70.1
traitlets==5.16.1
transformers==4.49.0
trimesh==4.7.4
triton==3.0.0
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.5
urllib3==2.8.0
wcwidth==0.9.1
Werkzeug==3.1.9
widgetsnbextension==4.0.16
yacs==0.1.8
yapf==0.43.0
zipp==4.1.1
```
