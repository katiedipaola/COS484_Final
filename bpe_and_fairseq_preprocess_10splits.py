import os
import random
import sentencepiece as spm

# Activate virtual environment 
os.system("source fairseq_env/bin/activate")

# Check Python version (optional)
os.system("python --version")

# Load original training set paths
SRC_PATH = "fairseq/examples/translation/iwslt14.tokenized.de-en/train.de"
TGT_PATH = "fairseq/examples/translation/iwslt14.tokenized.de-en/train.en"

OUTPUT_DIR = "data/iwslt14_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SPLITS = 10
SAMPLES_PER_SPLIT = 101_000

# Load original training set
with open(SRC_PATH, "r", encoding="utf-8") as f:
    src_lines = f.readlines()
with open(TGT_PATH, "r", encoding="utf-8") as f:
    tgt_lines = f.readlines()

assert len(src_lines) == len(tgt_lines), "Mismatched line counts."

dataset = list(zip(src_lines, tgt_lines))
total_samples = len(dataset)
print(f"Total samples: {total_samples}")

# Generate 10 random subsets
for i in range(NUM_SPLITS):
    subset = random.sample(dataset, SAMPLES_PER_SPLIT)
    split_src = os.path.join(OUTPUT_DIR, f"train{i}.de")
    split_tgt = os.path.join(OUTPUT_DIR, f"train{i}.en")
    with open(split_src, "w", encoding="utf-8") as f_src, open(split_tgt, "w", encoding="utf-8") as f_tgt:
        for src, tgt in subset:
            f_src.write(src)
            f_tgt.write(tgt)
    print(f"Saved split {i}: {split_src}, {split_tgt}")

# Install SentencePiece for BPE
os.system("pip install sentencepiece")

# Combine original training corpus (optional if needed for BPE)
os.system("cat fairseq/examples/translation/iwslt14.tokenized.de-en/train.de fairseq/examples/translation/iwslt14.tokenized.de-en/train.en > full_corpus.txt")

# Train BPE with 10K vocab
spm.SentencePieceTrainer.Train('--input=full_corpus.txt --model_prefix=bpe --vocab_size=10000 --character_coverage=1.0 --model_type=bpe')

# Load the model
sp = spm.SentencePieceProcessor(model_file='bpe.model')

for i in range(NUM_SPLITS):
    os.makedirs(f"data/iwslt14_bpe/train{i}", exist_ok=True)

    # Read source and target files
    with open(f"data/iwslt14_splits/train{i}.de", "r", encoding="utf-8") as f:
        de_lines = f.readlines()

    with open(f"data/iwslt14_splits/train{i}.en", "r", encoding="utf-8") as f:
        en_lines = f.readlines()

    # Encode using BPE
    de_encoded = [ " ".join(sp.encode(line.strip(), out_type=str)) + "\n" for line in de_lines ]
    en_encoded = [ " ".join(sp.encode(line.strip(), out_type=str)) + "\n" for line in en_lines ]

    # Save the encoded lines
    with open(f"data/iwslt14_bpe/train{i}/train.de", "w", encoding="utf-8") as f:
        f.writelines(de_encoded)

    with open(f"data/iwslt14_bpe/train{i}/train.en", "w", encoding="utf-8") as f:
        f.writelines(en_encoded)

# Preprocess all 10 splits using fairseq-preprocess
for i in range(NUM_SPLITS):
    print(f"Preprocessing split {i}...")

    os.system(f"fairseq-preprocess --source-lang de --target-lang en "
              f"--trainpref data/iwslt14_bpe/train{i}/train "
              f"--destdir data-bin/iwslt14_bpe_split{i} "
              f"--workers 2")

    print(f"Preprocessed split {i}: data-bin/iwslt14_bpe_split{i}")
