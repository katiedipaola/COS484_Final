import torch
import torch.serialization
from argparse import Namespace

# Allow loading Namespace if any remains in subprocess (precaution)
torch.serialization.add_safe_globals([Namespace])

import subprocess
import os


def strip_namespace(obj):
    """Recursively convert all argparse.Namespace objects to plain dicts."""
    if isinstance(obj, Namespace):
        return vars(obj)
    elif isinstance(obj, dict):
        return {k: strip_namespace(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_namespace(i) for i in obj]
    return obj


# Paths
bpe_codes = "bpe.codes"
input_file = "perturbed_inputs.de"
bpe_input_file = "perturbed_inputs.bpe.de"
dummy_target_file = "perturbed_inputs.en"
data_bin_dir = "data-bin/perturbed_bpe"
dict_src = "data-bin/iwslt14_bpe_full/dict.de.txt"
dict_tgt = "data-bin/iwslt14_bpe_full/dict.en.txt"
results_dir = "results/perturbed"

# -------------------------------
# Step 1: Apply BPE
print("Applying BPE to perturbed inputs...")
subprocess.run([
    "subword-nmt", "apply-bpe",
    "--codes", bpe_codes,
    "--input", input_file,
    "--output", bpe_input_file
], check=True)

# -------------------------------
# Step 2: Create dummy .en file
print("Creating dummy target file...")
num_lines = sum(1 for _ in open(bpe_input_file))
with open(dummy_target_file, "w") as f:
    for _ in range(num_lines):
        f.write("<dummy>\n")

# -------------------------------
# Step 3: Preprocess with fairseq
print("Running fairseq-preprocess...")
os.makedirs(data_bin_dir, exist_ok=True)
subprocess.run([
    "fairseq-preprocess",
    "--source-lang", "de",
    "--target-lang", "en",
    "--only-source",
    "--testpref", bpe_input_file.replace(".de", ""),
    "--destdir", data_bin_dir,
    "--srcdict", dict_src
], check=True)

# Step 3.5: Copy target dictionary (required even for source-only generation)
print("Copying target dictionary...")
subprocess.run(["cp", dict_tgt, os.path.join(data_bin_dir, "dict.en.txt")], check=True)

# -------------------------------
# Step 4: Generate predictions
print("Starting perturbed generation...")
os.makedirs(results_dir, exist_ok=True)

for i in range(4):
    print(f"\n Generating predictions for model {i}...")

    # Sanitize checkpoint
    orig_path = f"checkpoints/split{i}/checkpoint1.pt"
    safe_path = f"checkpoints/split{i}/checkpoint1_safe.pt"

    ckpt = torch.load(orig_path, weights_only=False)
    ckpt = strip_namespace(ckpt)
    torch.save(ckpt, safe_path)

    output_path = f"{results_dir}/model{i}"
    os.makedirs(output_path, exist_ok=True)

    command = [
        "fairseq-generate",
        data_bin_dir,
        "--path", safe_path,
        "--gen-subset", "test",
        "--beam", "5",
        "--remove-bpe",
        "--batch-size", "32",
        "--results-path", output_path
    ]

    subprocess.run(command, check=True)

print("\n✅ All perturbed predictions generated!")
