# Load the 10 sets of training indices
# Each indices{i}.txt file contains the line numbers (sentence IDs) used to train model i
train_sets = []
for i in range(10):
    with open(f"data/iwslt14_splits/indices{i}.txt") as f:
        indices = set(map(int, f.readlines()))
        train_sets.append(indices)

# Finding h←A(S\i)
from collections import defaultdict

# For each training example (0 to 159999), track which models didn't see it
sample_to_excluded_models = defaultdict(list)
for i in range(160000):  # Full training set
    for model_id, indices in enumerate(train_sets):
        if i not in indices:
            sample_to_excluded_models[i].append(model_id)

# Filter models with fewer than 2 exclusions so you have enough scores to average
valid_samples = {i for i, models in sample_to_excluded_models.items() if len(models) >= 2}
print(f"{len(valid_samples)} samples can be used for memorization scoring")

# Match predictions to the reference and score with chrf
from torchmetrics.text import CHRFScore
import torch
# Prepares the chrF metric to score predictions
chrf = CHRFScore(return_sentence_level_score=True)
memorization_scores = {}

# Load reference file once
with open("fairseq/examples/translation/iwslt14.tokenized.de-en/train.en") as ref_file:
    all_refs = ref_file.readlines()

# For each sample i ∈ valid_samples
for i in valid_samples:
    excluded_models = sample_to_excluded_models[i]
    included_models = [model_id for model_id in range(10) if model_id not in excluded_models]

    preds_excluded = []
    preds_included = []

    # Reference sentence is the same for both models 
    ref = all_refs[i].strip()

    # Get predictions from models that did NOT see i
    for model_id in excluded_models:
        with open(f"results/model{model_id}/generate-train.txt") as f:
            for line in f:
                if line.startswith(f"H-{i}\t"):
                    pred = line.strip().split('\t')[2]
                    preds_excluded.append(pred)
                    break

    # Get predictions from models that DID see i
    for model_id in included_models:
        with open(f"results/model{model_id}/generate-train.txt") as f:
            for line in f:
                if line.startswith(f"H-{i}\t"):
                    pred = line.strip().split('\t')[2]
                    preds_included.append(pred)
                    break

    # Compute memorization score only if we have predictions from both groups
    if preds_excluded and preds_included:
        # Prepare reference format (must be list of list of strings)
        targets_excluded = [[ref] for _ in preds_excluded]
        targets_included = [[ref] for _ in preds_included]

        # Use CHRFScore to get sentence-level scores
        scores_excluded = chrf(preds_excluded, targets_excluded).tolist()
        scores_included = chrf(preds_included, targets_included).tolist()

        avg_excluded = sum(scores_excluded) / len(scores_excluded)
        avg_included = sum(scores_included) / len(scores_included)


        memorization_scores[i] = avg_included - avg_excluded
    
# Save the memorization scores
import json

with open("memorization_scores.json", "w") as f:
    json.dump(memorization_scores, f)
