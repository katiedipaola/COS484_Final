import subprocess

for i in range(10):
    print(f"\n🚀 Training model on split {i}...\n")

    command = [
        "fairseq-train", f"data-bin/iwslt14_bpe_split{i}",
        "--arch", "transformer",
        "--encoder-layers", "6", "--decoder-layers", "6",
        "--encoder-embed-dim", "512", "--decoder-embed-dim", "512",
        "--encoder-ffn-embed-dim", "1024", "--decoder-ffn-embed-dim", "1024",
        "--encoder-attention-heads", "4", "--decoder-attention-heads", "4",
        "--share-decoder-input-output-embed",
        "--optimizer", "adam", "--adam-betas", "(0.9, 0.98)", "--clip-norm", "0.0",
        "--lr", "5e-4", "--lr-scheduler", "inverse_sqrt", "--warmup-updates", "4000",
        "--dropout", "0.3", "--criterion", "label_smoothed_cross_entropy", "--label-smoothing", "0.1",
        "--max-tokens", "4096",
        "--eval-bleu", "--eval-bleu-detok", "moses", "--eval-bleu-remove-bpe",
        "--best-checkpoint-metric", "bleu", "--maximize-best-checkpoint-metric",
        "--save-dir", f"checkpoints/split{i}",
        "--log-format", "simple", "--log-interval", "10",
        "--disable-validation"
    ]

    subprocess.run(command)
