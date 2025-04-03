#!/bin/bash

# Create and activate a virtual environment
PYTHON_CMD=$(command -v python3.9)
$PYTHON_CMD -m venv fairseq_env
source fairseq_env/bin/activate

# Upgrade pip to version 24
pip install --upgrade pip==24.0
pip install torch
pip install "numpy<2.0" 

# Ensure wget is installed
if ! command -v wget &> /dev/null; then
    echo "wget not found, installing..."
    brew install wget
fi

# Clone the Fairseq repository
git clone https://github.com/pytorch/fairseq
cd fairseq || exit

# Install Fairseq in editable mode
pip install --editable ./

CFLAGS="-stdlib=libc++" pip install --editable ./

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./

pip install pyarrow
pip install subword-nmt
pip install sacremoses
