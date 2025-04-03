import os
import random
from pathlib import Path
from shutil import copyfile

# === CONFIG ===
SRC_PATH = "fairseq/examples/translation/iwslt14.tokenized.de-en/train.de"
TGT_PATH = "fairseq/examples/translation/iwslt14.tokenized.de-en/train.en"
OUTPUT_DIR = Path("data/iwslt14_splits")
BPE_DIR = Path("data/iwslt14_bpe")
NUM_SPLITS = 10
SAMPLES_PER_SPLIT = 101_000
BPE_CODES = "bpe.codes"
BPE_VOCAB_SIZE = 10000

# === STEP 1: Load and shuffle dataset ===
with open(SRC_PATH, encoding="utf-8") as f:
    src_lines = f.readlines()
with open(TGT_PATH, encoding="utf-8") as f:
    tgt_lines = f.readlines()

assert len(src_lines) == len(tgt_lines), "Mismatch in line counts."
dataset = list(zip(src_lines, tgt_lines))

# === STEP 2: Write 10 random 101k-sample subsets ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for i in range(NUM_SPLITS):
    subset = random.sample(dataset, SAMPLES_PER_SPLIT)
    with open(OUTPUT_DIR / f"train{i}.de", "w", encoding="utf-8") as f_de, \
         open(OUTPUT_DIR / f"train{i}.en", "w", encoding="utf-8") as f_en:
        for src, tgt in subset:
            f_de.write(src.strip() + "\n")
            f_en.write(tgt.strip() + "\n")

# === STEP 3: Learn joint BPE ===
with open("full_corpus.txt", "w", encoding="utf-8") as f:
    for src, tgt in dataset:
        f.write(src.strip() + "\n")
        f.write(tgt.strip() + "\n")

os.system(f"subword-nmt learn-bpe -s {BPE_VOCAB_SIZE} < full_corpus.txt > {BPE_CODES}")

# === STEP 4: Apply BPE to all splits ===
BPE_DIR.mkdir(parents=True, exist_ok=True)
for i in range(NUM_SPLITS):
    split_bpe_dir = BPE_DIR / f"train{i}"
    split_bpe_dir.mkdir(parents=True, exist_ok=True)

    os.system(f"subword-nmt apply-bpe -c {BPE_CODES} < {OUTPUT_DIR}/train{i}.de > {split_bpe_dir}/train.de")
    os.system(f"subword-nmt apply-bpe -c {BPE_CODES} < {OUTPUT_DIR}/train{i}.en > {split_bpe_dir}/train.en")

# === STEP 5: fairseq-preprocess ===
for i in range(NUM_SPLITS):
    print(f"Preprocessing split {i}")
    os.system(
        f"fairseq-preprocess --source-lang de --target-lang en "
        f"--trainpref {BPE_DIR}/train{i}/train "
        f"--destdir data-bin/iwslt14_bpe_split{i} "
        f"--workers 2"
    )
