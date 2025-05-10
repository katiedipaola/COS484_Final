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

cd fairseq
mkdir -p bin
ln -s fairseq/fairseq_cli/fairseq-preprocess fairseq-preprocess
chmod +x bin/fairseq-preprocess

bash fairseq/examples/translation/prepare-iwslt14.sh

find fairseq -name "train.de"
ls fairseq/iwslt14.tokenized.de-en/

mkdir -p fairseq/examples/translation/iwslt14.tokenized.de-en/
cp fairseq/iwslt14.tokenized.de-en/train.de fairseq/examples/translation/iwslt14.tokenized.de-en/train.de
cp fairseq/iwslt14.tokenized.de-en/train.en fairseq/examples/translation/iwslt14.tokenized.de-en/train.en

ls fairseq/examples/translation/iwslt14.tokenized.de-en/train.de
