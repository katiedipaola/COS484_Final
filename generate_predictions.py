import subprocess

for i in range(10):
    print(f"Generating predictions for model {i}...")

    command = [
        "fairseq-generate",
        f"data-bin/iwslt14_bpe_full",
        "--path", f"checkpoints/split{i}/checkpoint70.pt",
        "--gen-subset", "train",
        "--beam", "5",
        "--remove-bpe",
        "--batch-size", "4096",
        "--results-path", f"results/model{i}"
    ]

    subprocess.run(command)

# generate predictions for the full model used in algorithm 2
command = [
        "fairseq-generate",
        f"data-bin/iwslt14_bpe_full",
        "--path", f"checkpoints/full_model/checkpoint70.pt",
        "--gen-subset", "train",
        "--beam", "5",
        "--remove-bpe",
        "--batch-size", "4096",
        "--results-path", f"results/full_model"
]
subprocess.run(command)
