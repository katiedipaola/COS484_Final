#!/bin/bash

# conda activate fairseq-gcc9
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
pip install pip==23.2.1
pip install fairseq
pip install subword-nmt torchmetrics

git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable . --verbose
pip install subword-nmt sacremoses

cd /scratch/network/kd5846/COS484_Final/fairseq
mkdir -p /scratch/network/kd5846/bin
ln -s /scratch/network/kd5846/COS484_Final/fairseq/fairseq_cli/fairseq-preprocess /scratch/network/kd5846/bin/fairseq-preprocess
chmod +x /scratch/network/kd5846/bin/fairseq-preprocess

bash /scratch/network/kd5846/COS484_Final/fairseq/examples/translation/prepare-iwslt14.sh


# # Create and activate a virtual environment
# PYTHON_CMD=$(command -v python3.9)

# if [ -z "$PYTHON_CMD" ]; then
#   echo "python3.9 not found. Please install it or use pyenv."
#   exit 1
# fi

# $PYTHON_CMD -m venv fairseq_env
# source fairseq_env/bin/activate

# # Upgrade pip to version 24
# pip install --upgrade pip==24.0
# pip install torch
# pip install "numpy<2.0" 
# pip install packaging
# pip install pyarrow subword-nmt sacremoses

# # Set CUDA_HOME to the correct location
# export CUDA_HOME="/usr/local/cuda"  # Replace with the actual path, e.g., /usr/local/cuda or $HOME/cuda

# # Update the PATH to include CUDA bin and lib64 directories
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# # # Ensure wget is installed
# # if ! command -v wget &> /dev/null; then
# #     echo "wget not found, installing..."
# #     brew install wget
# # fi

# # Clone the Fairseq repository
# git clone https://github.com/pytorch/fairseq
# cd fairseq
# CFLAGS='-std=c++11' pip install .

# # CFLAGS="-stdlib=libc++" pip install --editable ./

# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
#   --global-option="--deprecated_fused_adam" --global-option="--xentropy" --no-build-isolation\
#   --global-option="--fast_multihead_attn" ./

# pip install pyarrow
# pip install subword-nmt
# pip install sacremoses
