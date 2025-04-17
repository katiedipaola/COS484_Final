import subprocess
import os

perturbed_input_file = "perturbed_inputs.de"  
perturbed_results_dir = "results/perturbed"

os.makedirs(perturbed_results_dir, exist_ok=True)

for i in range(10):
    print(f"Generating perturbed predictions for model {i}...")

    model_path = f"checkpoints/split{i}/checkpoint1.pt"
    output_dir = f"{perturbed_results_dir}/model{i}"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "fairseq-generate",
        "data-bin/iwslt14_bpe_full",
        "--path", model_path,
        "--input", perturbed_input_file,  
        "--source-lang", "de", "--target-lang", "en",
        "--beam", "5",
        "--remove-bpe",
        "--batch-size", "32",
        "--results-path", output_dir
    ]

    subprocess.run(command)
