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

find /scratch/network/kd5846/COS484_Final/fairseq -name "train.de"
ls /scratch/network/kd5846/COS484_Final/fairseq/iwslt14.tokenized.de-en/

mkdir -p /scratch/network/kd5846/COS484_Final/fairseq/examples/translation/iwslt14.tokenized.de-en/
cp /scratch/network/kd5846/COS484_Final/fairseq/iwslt14.tokenized.de-en/train.de /scratch/network/kd5846/COS484_Final/fairseq/examples/translation/iwslt14.tokenized.de-en/train.de
cp /scratch/network/kd5846/COS484_Final/fairseq/iwslt14.tokenized.de-en/train.en /scratch/network/kd5846/COS484_Final/fairseq/examples/translation/iwslt14.tokenized.de-en/train.en

ls /scratch/network/kd5846/COS484_Final/fairseq/examples/translation/iwslt14.tokenized.de-en/train.de
