# Step 1: Apply BPE to perturbed inputs
print("Applying BPE to perturbed inputs...")
subprocess.run([
    "subword-nmt", "apply-bpe", 
    "--codes", "bpe.codes",
    "--input", "perturbed_inputs.de",
    "--output", "perturbed_inputs.bpe.de"
])

# -------------------------------
# Step 2: Create dummy .en file for Fairseq
print("Creating dummy target file...")
num_lines = sum(1 for _ in open("perturbed_inputs.bpe.de"))
with open("perturbed_inputs.en", "w") as f:
    for _ in range(num_lines):
        f.write("<dummy>\n")

# -------------------------------
# Step 3: Preprocess with fairseq-preprocess
print("Running fairseq-preprocess...")
os.makedirs("data-bin/perturbed_bpe", exist_ok=True)

subprocess.run([
    "fairseq-preprocess",
    "--source-lang", "de",
    "--target-lang", "en",
    "--only-source",
    "--testpref", "perturbed_inputs.bpe",
    "--destdir", "data-bin/perturbed_bpe",
    "--srcdict", "data-bin/iwslt14_bpe_full/dict.de.txt"
])

# -------------------------------
# Step 4: Generate predictions from each model
print("Starting perturbed generation...")

perturbed_results_dir = "results/perturbed"
os.makedirs(perturbed_results_dir, exist_ok=True)

for i in range(10):
    print(f"\n🌀 Generating predictions for model {i}...")

    model_path = f"checkpoints/split{i}/checkpoint1.pt"
    output_dir = f"{perturbed_results_dir}/model{i}"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "fairseq-generate",
        "data-bin/perturbed_bpe",
        "--path", model_path,
        "--source-lang", "de",
        "--target-lang", "en",
        "--gen-subset", "test",
        "--beam", "5",
        "--remove-bpe",
        "--batch-size", "32",
        "--results-path", output_dir
    ]

    subprocess.run(command)

print("All perturbed predictions generated!")
