#!/usr/bin/env bash
set -e

# do not run in base
if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  echo "activate a non-base conda env and re-run."
  exit 1
fi

mamba install -y pip

# scientific stl
mamba install -y \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    scikit-learn \
    scipy \
    ipykernel \
    jupyter

# kernel spec for Jupyter
python3 -m ipykernel install --user \
    --name="pyg_CUDA_py311" \
    --display-name="pyg_CUDA (py3.11)"

# torch 2.1.0 + cuda 11.8 with COMPATIBLE torchvision version
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# matching torch 2.1.0+cu118 for PyTorch Geometric
python3 -m pip install torch-geometric==2.4.0 --no-cache-dir
python3 -m pip install \
    pyg-lib==0.2.0+cu118 \
    torch-scatter==2.2.1+cu118 \
    torch-sparse==0.6.17+cu118 \
    torch-cluster==1.6.0+cu118 \
    torch-spline-conv==1.2.1+cu118 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# smoke test
python3 - <<'PYTEST'
import torch, torch_geometric
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("Torch‑Geometric:", torch_geometric.__version__)
PYTES