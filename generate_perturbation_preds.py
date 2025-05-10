import os
import subprocess
import torch
import torch.serialization
from argparse import Namespace

# Allow loading Namespace safely
torch.serialization.add_safe_globals([Namespace])

def strip_namespace(obj):
    if isinstance(obj, Namespace):
        return vars(obj)
    elif isinstance(obj, dict):
        return {k: strip_namespace(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_namespace(i) for i in obj]
    return obj

# === Paths ===
bpe_codes = "bpe.codes"
input_file = "perturbed_inputs.de"
bpe_input_file = "perturbed_inputs.bpe.de"
dummy_target_file = "perturbed_inputs.en"
data_bin_dir = "data-bin/perturbed_bpe"
dict_src = "data-bin/iwslt14_bpe_full/dict.de.txt"
dict_tgt = "data-bin/iwslt14_bpe_full/dict.en.txt"
results_dir = "results/perturbed/full_model"
model_path = "checkpoints/full_model/checkpoint70.pt"

# === Step 1: Apply BPE ===
print("Applying BPE to perturbed inputs...")
subprocess.run([
    "subword-nmt", "apply-bpe",
    "--codes", bpe_codes,
    "--input", input_file,
    "--output", bpe_input_file
], check=True)

# === Step 2: Dummy target ===
print("Creating dummy target file...")
num_lines = sum(1 for _ in open(bpe_input_file))
with open(dummy_target_file, "w") as f:
    for _ in range(num_lines):
        f.write("<dummy>\n")

# === Step 3: fairseq-preprocess ===
print("Running fairseq-preprocess for perturbed inputs...")
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

print("Copying target dictionary...")
subprocess.run(["cp", dict_tgt, os.path.join(data_bin_dir, "dict.en.txt")], check=True)

# === Step 4: Sanitize checkpoint (remove Namespace) ===
print("Stripping Namespace from model checkpoint...")
safe_model_path = model_path.replace(".pt", "_safe.pt")
ckpt = torch.load(model_path, weights_only=False)
ckpt = strip_namespace(ckpt)
torch.save(ckpt, safe_model_path)

# === Step 5: Generate predictions ===
print("Generating predictions on perturbed inputs...")
os.makedirs(results_dir, exist_ok=True)
subprocess.run([
    "fairseq-generate",
    data_bin_dir,
    "--path", safe_model_path,
    "--gen-subset", "test",
    "--beam", "5",
    "--remove-bpe",
    "--batch-size", "4096",
    "--results-path", results_dir
], check=True)

print("✅ Finished generating perturbed predictions.")
