#!/bin/bash

fairseq-generate fairseq/data-bin/iwslt14.tokenized.de-en \
    --path fairseq/checkpoints/fconv/checkpoint_best.pt \
    --batch-size 128 --beam 5