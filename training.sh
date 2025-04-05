#!/bin/bash

# train model
source fairseq_env/bin/activate
cd fairseq

mkdir -p checkpoints/fconv
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
    --optimizer nag --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-epoch 70 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir checkpoints/fconv