import json
from glob import glob

combined = {}
for f in glob("memorization_scores_chunk*.json"):
    with open(f) as file:
        combined.update(json.load(file))

with open("memorization_scores.json", "w") as f:
    json.dump(combined, f)

print(f"Combined {len(combined)} scores into memorization_scores.json")

