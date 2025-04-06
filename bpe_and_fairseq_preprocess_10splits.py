import os
import random
from pathlib import Path
from shutil import copyfile

SRC_PATH = "fairseq/examples/translation/iwslt14.tokenized.de-en/train.de"
TGT_PATH = "fairseq/examples/translation/iwslt14.tokenized.de-en/train.en"
OUTPUT_DIR = Path("data/iwslt14_splits")
BPE_DIR = Path("data/iwslt14_bpe")
NUM_SPLITS = 10
SAMPLES_PER_SPLIT = 101_000
BPE_CODES = "bpe.codes"
BPE_VOCAB_SIZE = 10000

# Load and shuffle the dataset
with open(SRC_PATH, encoding="utf-8") as f:
    src_lines = f.readlines()
with open(TGT_PATH, encoding="utf-8") as f:
    tgt_lines = f.readlines()

assert len(src_lines) == len(tgt_lines), "Mismatch in line counts."
dataset = list(zip(src_lines, tgt_lines))

# Create the random subsets
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for i in range(NUM_SPLITS):
    indices = random.sample(range(len(dataset)), SAMPLES_PER_SPLIT)
    with open(OUTPUT_DIR / f"train{i}.de", "w", encoding="utf-8") as f_de, \
         open(OUTPUT_DIR / f"train{i}.en", "w", encoding="utf-8") as f_en, \
         open(OUTPUT_DIR / f"indices{i}.txt", "w", encoding="utf-8") as f_idx:
             
        for idx in indices:
            src, tgt = dataset[idx]
            f_de.write(src.strip() + "\n")
            f_en.write(tgt.strip() + "\n")
            f_idx.write(str(idx) + "\n")

# Learn joint BPE from the full dataset
with open("full_corpus.txt", "w", encoding="utf-8") as f:
    for src, tgt in dataset:
        f.write(src.strip() + "\n")
        f.write(tgt.strip() + "\n")

os.system(f"subword-nmt learn-bpe -s {BPE_VOCAB_SIZE} < full_corpus.txt > {BPE_CODES}")

# Apply BPE to all splits 
BPE_DIR.mkdir(parents=True, exist_ok=True)
for i in range(NUM_SPLITS):
    split_bpe_dir = BPE_DIR / f"train{i}"
    split_bpe_dir.mkdir(parents=True, exist_ok=True)

    os.system(f"subword-nmt apply-bpe -c {BPE_CODES} < {OUTPUT_DIR}/train{i}.de > {split_bpe_dir}/train.de")
    os.system(f"subword-nmt apply-bpe -c {BPE_CODES} < {OUTPUT_DIR}/train{i}.en > {split_bpe_dir}/train.en")

# Build shared vocabulary from full dataset
SHARED_VOCAB_DIR = Path("data-bin/iwslt14_bpe_shared_vocab")
SHARED_VOCAB_DIR.mkdir(parents=True, exist_ok=True)

# Save full dataset 
FULL_BPE_DIR = Path("data/iwslt14_bpe_full")
FULL_BPE_DIR.mkdir(parents=True, exist_ok=True)

# Save original files as BPE-applied
with open(FULL_BPE_DIR / "train.de", "w", encoding="utf-8") as f_de, \
     open(FULL_BPE_DIR / "train.en", "w", encoding="utf-8") as f_en:
    for src, tgt in dataset:
        f_de.write(src.strip() + "\n")
        f_en.write(tgt.strip() + "\n")

# Apply BPE to full dataset
os.system(f"subword-nmt apply-bpe -c {BPE_CODES} < {FULL_BPE_DIR}/train.de > {FULL_BPE_DIR}/bpe.de")
os.system(f"subword-nmt apply-bpe -c {BPE_CODES} < {FULL_BPE_DIR}/train.en > {FULL_BPE_DIR}/bpe.en")

# Preprocess full BPE data to generate shared vocab
os.system(
    f"fairseq-preprocess --source-lang de --target-lang en "
    f"--trainpref {FULL_BPE_DIR}/bpe "
    f"--destdir {SHARED_VOCAB_DIR} "
    f"--workers 2"
)

# fairseq-preprocess all splits with shared vocab
for i in range(NUM_SPLITS):
    print(f"Preprocessing split {i}")
    os.system(
        f"fairseq-preprocess --source-lang de --target-lang en "
        f"--trainpref {BPE_DIR}/train{i}/train "
        f"--destdir data-bin/iwslt14_bpe_split{i} "
        f"--workers 2"
    )
    os.system(
        f"cp {SHARED_VOCAB_DIR}/dict.de.txt data-bin/iwslt14_bpe_split{i}/ && "
        f"cp {SHARED_VOCAB_DIR}/dict.en.txt data-bin/iwslt14_bpe_split{i}/"
    )
# Prepare full dataset for evaluation with shared vocab
EVAL_BPE_DIR = Path("data/iwslt14_bpe_eval")
EVAL_BPE_DIR.mkdir(parents=True, exist_ok=True)

with open(EVAL_BPE_DIR / "train.de", "w", encoding="utf-8") as f_de, \
     open(EVAL_BPE_DIR / "train.en", "w", encoding="utf-8") as f_en:
    for src, tgt in dataset:
        f_de.write(src.strip() + "\n")
        f_en.write(tgt.strip() + "\n")

os.system(f"subword-nmt apply-bpe -c {BPE_CODES} < {EVAL_BPE_DIR}/train.de > {EVAL_BPE_DIR}/bpe.de")
os.system(f"subword-nmt apply-bpe -c {BPE_CODES} < {EVAL_BPE_DIR}/train.en > {EVAL_BPE_DIR}/bpe.en")

os.system(
    f"fairseq-preprocess --source-lang de --target-lang en "
    f"--trainpref {EVAL_BPE_DIR}/bpe "
    f"--destdir data-bin/iwslt14_bpe_full "
    f"--workers 2"
)

os.system(
    f"cp {SHARED_VOCAB_DIR}/dict.de.txt data-bin/iwslt14_bpe_full/ && "
    f"cp {SHARED_VOCAB_DIR}/dict.en.txt data-bin/iwslt14_bpe_full/"
)
