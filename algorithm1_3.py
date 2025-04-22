import argparse
import json
from collections import defaultdict
from torchmetrics.text import CHRFScore, BLEUScore
from math import ceil
from multiprocessing import Pool, cpu_count

# --- Argparse for SLURM array inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--array-task-id", type=int, required=True)
parser.add_argument("--num-tasks", type=int, required=True)
args = parser.parse_args()

task_id = args.array_task_id
num_tasks = args.num_tasks

# --- Load training indices ---
train_sets = [set(map(int, open(f"data/iwslt14_splits/indices{i}.txt"))) for i in range(10)]

# --- Map sample to excluded models ---
sample_to_excluded_models = defaultdict(list)
for model_id, indices in enumerate(train_sets):
    for i in range(160000):
        if i not in indices:
            sample_to_excluded_models[i].append(model_id)

# --- Valid samples only ---
valid_samples = [i for i, models in sample_to_excluded_models.items() if len(models) >= 3]
valid_samples.sort()

# --- Split sample list into chunks ---
chunk_size = ceil(len(valid_samples) / num_tasks)
start = task_id * chunk_size
end = min((task_id + 1) * chunk_size, len(valid_samples))
chunk = valid_samples[start:end]

# --- Load references ---
with open("fairseq/examples/translation/iwslt14.tokenized.de-en/train.en") as f:
    all_refs = [line.strip() for line in f]

# --- Load model predictions ---
model_preds = [{} for _ in range(10)]
for model_id in range(10):
    with open(f"results/model{model_id}/generate-train.txt") as f:
        for line in f:
            if line.startswith("H-"):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    sample_id = int(parts[0][2:])
                    model_preds[model_id][sample_id] = parts[2]

# --- Global shared data ---
bleu = BLEUScore(n_gram=1)

def score_sample(i):
    ref = all_refs[i]
    excluded = sample_to_excluded_models[i]
    included = [m for m in range(10) if m not in excluded]

    preds_excluded = [model_preds[m][i] for m in excluded if i in model_preds[m]]
    preds_included = [model_preds[m][i] for m in included if i in model_preds[m]]

    if not preds_excluded or not preds_included:
        return None

    targets_excluded = [[ref]] * len(preds_excluded)
    targets_included = [[ref]] * len(preds_included)

    scores_excluded_bleu = bleu(preds_excluded, targets_excluded)
    scores_included_bleu = bleu(preds_included, targets_included)
    bleu_score = (scores_included_bleu.mean() - scores_excluded_bleu.mean()).item()   

    return (i, bleu_score)

# --- Multiprocessing scoring ---
if __name__ == "__main__":
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(score_sample, chunk)

    results = [r for r in results if r is not None]

    # Collect and save results into dictionaries
    memorization_scores_bleu = {i: bleu for (i, bleu) in results}

    with open(f"memorization_scores_bleu_{task_id}_excl1.json", "w") as f:
        json.dump(memorization_scores_bleu, f)
    print(f"Saved BLEU scores to memorization_scores_bleu_{task_id}.json")

    if task_id == 0:
        with open("model_preds3.json", "w") as f:
                json.dump(model_preds, f)
            print("Saved: model_preds3.json")
