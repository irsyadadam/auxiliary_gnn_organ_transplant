#!/usr/bin/env bash
set -e

# do not run in base
if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  echo "activate a non-base conda env and re-run."
  exit 1
fi

python3 -m pip install --upgrade pip setuptools wheel

# Install NumPy first and pin it throughout
python -m pip install "numpy>=1.21.0,<1.25.0"

# Install PyTorch 2.1.0 + CUDA 11.8 with NumPy constraint
python -m pip install \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    "numpy>=1.21.0,<1.25.0" \
    --index-url https://download.pytorch.org/whl/cu118 \
    --no-cache-dir

# Install PyG with NumPy constraint
python -m pip install torch-geometric==2.4.0 "numpy>=1.21.0,<1.25.0" --no-cache-dir

# Fix the PyG extensions versions (0.2.0 doesn't exist for cu118)
python -m pip install \
    pyg-lib==0.3.1+pt21cu118 \
    torch-scatter==2.2.1+cu118 \
    torch-sparse==0.6.17+cu118 \
    torch-cluster==1.6.0+cu118 \
    torch-spline-conv==1.2.1+cu118 \
    "numpy>=1.21.0,<1.25.0" \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    --no-cache-dir

# Install other packages with NumPy constraint
python -m pip install \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    scipy \
    "numpy>=1.21.0,<1.25.0"