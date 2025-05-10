import json
from glob import glob

excl = "1"

combined = {}
for f in glob(f"memorization_scores_chrf_*_excl{excl}.json"):
    with open(f) as file:
        combined.update(json.load(file))
with open(f"memorization_scores_chrf_excl{excl}.json", "w") as f:        
    json.dump(combined, f)

combined = {}
for f in glob(f"memorization_scores_bleu_*_excl{excl}.json"):
    with open(f) as file:
        combined.update(json.load(file))

with open(f"memorization_scores_bleu_excl{excl}.json", "w") as f:
    json.dump(combined, f)

combined = {}
for f in glob(f"memorization_scores_acc_*_excl{excl}.json"):
    with open(f) as file:
        combined.update(json.load(file))

with open(f"memorization_scores_acc_excl{excl}.json", "w") as f:
    json.dump(combined, f)

print(f"Combined {len(combined)} scores into memorization_scores.json")

