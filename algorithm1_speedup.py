import json
from collections import defaultdict
from torchmetrics.text import CHRFScore, BLEUScore
import torch

# -------------------------------
# Load training indices once
train_sets = []
for i in range(10):
    with open(f"data/iwslt14_splits/indices{i}.txt") as f:
        train_sets.append(set(map(int, f)))

print("Loaded train sets")

# -------------------------------
# Map each sample to which models excluded it
sample_to_excluded_models = defaultdict(list)
for model_id, indices in enumerate(train_sets):
    for i in range(160000):
        if i not in indices:
            sample_to_excluded_models[i].append(model_id)

print("Tracked which models did not see examples")

# -------------------------------
# Keep only samples with 2+ excluded models
valid_samples = {i for i, models in sample_to_excluded_models.items() if len(models) >= 2}
print(f"{len(valid_samples)} samples can be used for memorization scoring")

# -------------------------------
# Load reference sentences
with open("fairseq/examples/translation/iwslt14.tokenized.de-en/train.en") as ref_file:
    all_refs = [line.strip() for line in ref_file]
print("Loaded reference file")

# -------------------------------
# Load all model predictions once
model_preds = [{} for _ in range(10)]  # List of dicts: model_preds[model_id][sample_id] = prediction
for model_id in range(10):
    with open(f"results/model{model_id}/generate-train.txt") as f:
        for line in f:
            if line.startswith("H-"):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    sample_id = int(parts[0][2:])  # H-1234
                    model_preds[model_id][sample_id] = parts[2]
print("Loaded model predictions")

# -------------------------------
# Score using chrF, bleu, accuracy
chrf = CHRFScore(return_sentence_level_score=True)
bleu = BLEUScore(n_gram=1)

memorization_scores_chrf = {}
memorization_scores_bleu = {}
memorization_scores_acc = {}

for i in valid_samples:
    excluded_models = sample_to_excluded_models[i]
    included_models = [m for m in range(10) if m not in excluded_models]
    ref = all_refs[i]

    preds_excluded = [model_preds[m].get(i) for m in excluded_models if i in model_preds[m]]
    preds_included = [model_preds[m].get(i) for m in included_models if i in model_preds[m]]

    if preds_excluded and preds_included:
        targets_excluded = [[ref]] * len(preds_excluded)
        targets_included = [[ref]] * len(preds_included)

        # CHRF
        _, scores_excluded = chrf(preds_excluded, targets_excluded)
        _, scores_included = chrf(preds_included, targets_included)
        memorization_scores_chrf[i] = (scores_included_chrf.mean() - scores_excluded_chrf.mean()).item()

        # BLEU
        scores_excluded_bleu = bleu(preds_excluded, targets_excluded)
        scores_included_bleu = bleu(preds_included, targets_included)
        memorization_scores_bleu[i] = (scores_included_bleu.mean() - scores_excluded_bleu.mean()).item()

        ref_tokens = ref.split()
        # Accuracy
        for p in preds_excluded:
          # Tokenize the reference and prediction
          pred_tokens = p.split()
        for p in preds_included:
          pred_tokens_i = p.split()

          # Count how many tokens are in both the prediction and reference
        common_tokens_exc = set(ref_tokens) & set(pred_tokens)
        common_tokens_inc = set(ref_tokens) & set(pred_tokens_i)
        acc_excluded = len(common_tokens_exc) / len(ref_tokens)
        acc_included = len(common_tokens_inc) / len(ref_tokens)

        memorization_scores_acc[i] = acc_included - acc_excluded     

print("Computed memorization scores")

# -------------------------------
# Save JSON output
with open("memorization_scores_chrf.json", "w") as f:
    json.dump(memorization_scores_chrf, f)
print("Memorization scores saved to 'memorization_scores_chrf.json'")

with open("memorization_scores_bleu.json", "w") as f:
    json.dump(memorization_scores_bleu, f)
print("Memorization scores saved to 'memorization_scores_bleu.json'")

with open("memorization_scores_acc.json", "w") as f:
    json.dump(memorization_scores_acc, f)
print("Memorization scores saved to 'memorization_scores_acc.json'")

with open("model_preds.json", "w") as f:
    json.dump(model_preds, f)
print("Saved: model_preds.json")
