import subprocess

for i in range(10):
    print(f"Generating predictions for model {i}...")

    command = [
        "fairseq-generate",
        f"data-bin/iwslt14_bpe_split{i}",
        "--path", f"checkpoints/split{i}/checkpoint1.pt",
        "--gen-subset", "train",
        "--beam", "5",
        "--remove-bpe",
        "--batch-size", "32",
        "--results-path", f"results/model{i}"
    ]

    subprocess.run(command)
