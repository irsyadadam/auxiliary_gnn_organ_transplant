#!/usr/bin/env bash
set -e

# do not run in base
if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  echo "activate a non-base conda env and re-run."
  exit 1
fi

# torch 2.1.0 + cuda 11.8 with COMPATIBLE torchvision version
python3 -m pip install --upgrade pip setuptools wheel

# Critical: Install NumPy first with compatible version
python -m pip install "numpy>=1.21.0,<1.25.0"

# Install PyTorch 2.1.0 + CUDA 11.8 (stable and well-tested)
python -m pip install \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --no-cache-dir

# Install PyG with matching PyTorch version
python -m pip install torch-geometric==2.4.0 --no-cache-dir

# Install PyG extensions for CUDA 11.8 + PyTorch 2.1.0
python -m pip install \
    pyg-lib==0.2.0+cu118 \
    torch-scatter==2.2.1+cu118 \
    torch-sparse==0.6.17+cu118 \
    torch-cluster==1.6.0+cu118 \
    torch-spline-conv==1.2.1+cu118 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    --no-cache-dir

python -m pip install \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    scipy