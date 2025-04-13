# memorization_chunk.py
import json, sys
from collections import defaultdict
from torchmetrics.text import CHRFScore
import numpy as np

chunk_id = int(sys.argv[1])
total_chunks = int(sys.argv[2])

# Load training sets
train_sets = []
for i in range(10):
    with open(f"data/iwslt14_splits/indices{i}.txt") as f:
        train_sets.append(set(map(int, f)))

sample_to_excluded_models = defaultdict(list)
for model_id, indices in enumerate(train_sets):
    for i in range(160000):
        if i not in indices:
            sample_to_excluded_models[i].append(model_id)

valid_samples = [i for i, models in sample_to_excluded_models.items() if len(models) >= 2]
valid_samples.sort()

# Split samples across chunks
chunk_size = len(valid_samples) // total_chunks
start = chunk_id * chunk_size
end = (chunk_id + 1) * chunk_size if chunk_id < total_chunks - 1 else len(valid_samples)
my_samples = valid_samples[start:end]

with open("fairseq/examples/translation/iwslt14.tokenized.de-en/train.en") as ref_file:
    all_refs = [line.strip() for line in ref_file]

model_preds = [{} for _ in range(10)]
for model_id in range(10):
    with open(f"results/model{model_id}/generate-train.txt") as f:
        for line in f:
            if line.startswith("H-"):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    sample_id = int(parts[0][2:])
                    model_preds[model_id][sample_id] = parts[2]

def compute_mem_score(i):
    excluded = sample_to_excluded_models[i]
    included = [m for m in range(10) if m not in excluded]
    ref = all_refs[i]

    preds_ex = [model_preds[m][i] for m in excluded if i in model_preds[m]]
    preds_in = [model_preds[m][i] for m in included if i in model_preds[m]]

    if not preds_ex or not preds_in:
        return None

    targets_ex = [[ref]] * len(preds_ex)
    targets_in = [[ref]] * len(preds_in)

    chrf = CHRFScore(return_sentence_level_score=True)
    _, scores_ex = chrf(preds_ex, targets_ex)
    _, scores_in = chrf(preds_in, targets_in)

    return i, float(np.mean(scores_in) - np.mean(scores_ex))

results = {}
for i in my_samples:
    res = compute_mem_score(i)
    if res:
        results[res[0]] = res[1]

with open(f"memorization_scores_chunk{chunk_id}.json", "w") as f:
    json.dump(results, f)

print(f"Chunk {chunk_id} complete: {len(results)} scores")

