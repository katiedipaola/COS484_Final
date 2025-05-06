def load_perturbed_predictions(filepath, line_map):
    preds = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("H-"):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    line_num = int(parts[0][2:])
                    y_tilde = parts[2]
                    if line_num < len(line_map):
                        idx, t = line_map[line_num]
                        preds[(idx, t)] = y_tilde
    return preds

perturbed_preds_full_model = load_perturbed_predictions(
    "results/perturbed/full_model/generate-test.txt", line_map)


def algorithm2_from_predictions(model_preds, perturbed_preds, parallel_corpus, indices, T, model_id):
    """
    Args:
        model_preds: dict of original predictions, e.g. model_preds[model_id][index] = y'
        perturbed_preds: dict of perturbed predictions, e.g. perturbed_preds[(index, token)] = y_tilde
        parallel_corpus: list of (src, tgt) sentence pairs
        indices: list of global dataset indices matching parallel_corpus
        T: list of perturbation tokens
        model_id: which model to read from (e.g., 0–9)

    Returns:
        H: list of hallucinated sample records (could also just be list of x if preferred)
    """
    H = []

    for i, (x, y) in enumerate(parallel_corpus):
        idx = indices[i]

        if idx not in model_preds:
            continue

        y_prime = model_preds[idx]
        if adjusted_bleu(y_prime, y.strip()) > 0.09:
            for t in T:
                perturbed_key = (idx, t)
                if perturbed_key not in perturbed_preds:
                    continue

                y_tilde = perturbed_preds[perturbed_key]
                if adjusted_bleu(y_tilde, y_prime) < 0.01:
                    H.append({
                        # "idx": idx,
                        "src": x.strip(),
                        # "tgt": y.strip(),
                        # "perturb_token": t,
                        # "y_prime": y_prime.strip(),
                        # "y_tilde": y_tilde.strip()
                    })
                    break  # only need one hallucination to count the sample

    return H

with open("model_preds.json") as f:
    model_preds = json.load(f)

H_mem_chrf = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_mem_chrf,
    indices=indices_mem_chrf,
    T=T,
    model_id=0
)

H_rand_chrf = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_rand_chrf,
    indices=indices_rand_chrf,
    T=T,
    model_id=0
)

H_mem_bleu = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_mem_bleu,
    indices=indices_mem_bleu,
    T=T,
    model_id=0
)

H_rand_bleu = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_rand_bleu,
    indices=indices_rand_bleu,
    T=T,
    model_id=0
)

H_mem_acc = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_mem_acc,
    indices=indices_mem_acc,
    T=T,
    model_id=0
)

H_rand_acc = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_rand_acc,
    indices=indices_rand_acc,
    T=T,
    model_id=0
)

def Unique(H):
  non_repeat = []
  np.sort(H)
  for i in range(len(H)):
    if i == 0:
      non_repeat.append(H[i])
    elif H[i] != H[i-1]:
      non_repeat.append(H[i])
  return non_repeat

unique_H_mem_chrf = Unique(H_mem_chrf)
unique_H_rand_chrf = Unique(H_rand_chrf)

unique_H_mem_bleu = Unique(H_mem_bleu)
unique_H_rand_bleu = Unique(H_rand_bleu)

unique_H_mem_acc = Unique(H_mem_acc)
unique_H_rand_acc = Unique(H_rand_acc)


# with BLEU scores onwards:
# data collection for Figure 3 (Top)
def topfiguredata(dictionary, threshold):
  sorted_dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
  sorted_dictionary_threshold = {k: v for k, v in sorted_dictionary.items() if v > threshold}
  print(sorted_dictionary)

  parallel_mem, _, indices, _ = ParallelCorpus(range_val=len(sorted_dictionar_threshold), sorted_dictionary=sorted_dictionary_threshold)
  H_mem = algorithm2_from_predictions(model_preds=model_preds,
  perturbed_preds=perturbed_preds_full_model,
  parallel_corpus= parallel_mem,
  indices=indices,
  T=T,
  model_id=0)
  H_mem_len = len(H_mem)
  unique_H_mem = Unique(H_mem)
  unique_H_mem_len = len(unique_H_mem)
  
  return H_mem_len, unique_H_mem_len

thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

total_mem_acc = [len(H_mem_acc)]
unique_mem_acc = [len(unique_H_mem_acc)]
total_mem_bleu = [len(H_mem_bleu)]
unique_mem_bleu = [len(unique_H_mem_acc)]
total_mem_chrf = [len(H_mem_chrf)]
unique_mem_chrf = [len(unique_H_mem_acc)]

from itertools import islice 

dictionary_acc_100 = dict(islice(dictionary_acc.items(),100))
dictionary_bleu_100 = dict(islice(dictionary_bleu.items(),100))
dictionary_chrf_100 = dict(islice(dictionary_chrf.items(),100))

for threshold in thresholds:
    mem_len_acc, unique_mem_len_acc = topfiguredata(dictionary_acc_100, threshold)
    mem_len_bleu, unique_mem_len_bleu = topfiguredata(dictionary_bleu_100, threshold)
    mem_len_chrf, unique_mem_len_chrf = topfiguredata(dictionary_chrf_100, threshold)

    total_mem_acc.append(mem_len_acc)
    unique_mem_acc.append(unique_mem_len_acc)
    total_mem_bleu.append(mem_len_bleu)
    unique_mem_bleu.append(unique_mem_len_bleu)
    total_mem_chrf.append(mem_len_chrf)
    unique_mem_chrf.append(unique_mem_len_chrf)


# graph for Figure 3 (Top) acc
import matplotlib.pyplot as plt
import numpy as np

labels = ['Top', '>0.9', '>0.8', '>0.7', '>0.6', '>0.5', '>0.4', '>0.3', '>0.2', '>0.1', '>0']
bar_width = 0.4

x = np.arange(len(labels))  # center positions

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, unique_mem_acc, width=bar_width, label='Unique HP', color='blue')
plt.bar(x + bar_width/2, total_mem_acc, width=bar_width, label='Total HP', color='orange')

plt.xticks(x, labels, rotation=45)
plt.xlabel('Memorization Values')
plt.ylabel('Hallucinations')
plt.title('Hallucinations vs Memorization Values for Accuracy Metric')
plt.legend()
plt.tight_layout()
plt.savefig("figures/hallucinations_vs_memorization_acc.png", dpi=300)
plt.show()

# graph for Figure 3 (Top) bleu

labels = ['Top', '>0.9', '>0.8', '>0.7', '>0.6', '>0.5', '>0.4', '>0.3', '>0.2', '>0.1', '>0']
bar_width = 0.4

x = np.arange(len(labels))  # center positions

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, unique_mem_bleu, width=bar_width, label='Unique HP', color='blue')
plt.bar(x + bar_width/2, total_mem_bleu, width=bar_width, label='Total HP', color='orange')

plt.xticks(x, labels, rotation=45)
plt.xlabel('Memorization Values')
plt.ylabel('Hallucinations')
plt.title('Hallucinations vs Memorization Values for Accuracy Metric')
plt.legend()
plt.tight_layout()
plt.savefig("figures/hallucinations_vs_memorization_acc.png", dpi=300)
plt.show()

# graph for Figure 3 (Top) chrf

labels = ['Top', '>0.9', '>0.8', '>0.7', '>0.6', '>0.5', '>0.4', '>0.3', '>0.2', '>0.1', '>0']
bar_width = 0.4

x = np.arange(len(labels))  # center positions

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, unique_mem_chrf, width=bar_width, label='Unique HP', color='blue')
plt.bar(x + bar_width/2, total_mem_chrf, width=bar_width, label='Total HP', color='orange')

plt.xticks(x, labels, rotation=45)
plt.xlabel('Memorization Values')
plt.ylabel('Hallucinations')
plt.title('Hallucinations vs Memorization Values for Accuracy Metric')
plt.legend()
plt.tight_layout()
plt.savefig("figures/hallucinations_vs_memorization_acc.png", dpi=300)
plt.show()



dictionary_bleu1 = sort_mems("memorization_scores_bleu1.json")
dictionary_bleu2 = sort_mems("memorization_scores_bleu.json")
dictionary_bleu3 = sort_mems("memorization_scores_bleu3.json")
dictionary_bleu4 = sort_mems("memorization_scores_bleu4.json")

dictionary_chrf1 = sort_mems("memorization_scores_chrf1.json")
dictionary_chrf2 = sort_mems("memorization_scores_chrf.json")
dictionary_chrf3 = sort_mems("memorization_scores_chrf3.json")
dictionary_chrf4 = sort_mems("memorization_scores_chrf4.json")

dictionary_acc1 = sort_mems("memorization_scores_acc1.json")
dictionary_acc2 = sort_mems("memorization_scores_accf.json")
dictionary_acc3 = sort_mems("memorization_scores_acc3.json")
dictionary_acc4 = sort_mems("memorization_scores_acc4.json")

def bottom_figure_data(ex_dicts_metric):
  exclusions = [1, 2, 3, 4]
  ex_dicts = ex_dicts_metric
  ex_rand = []
  ex_mem = []

  for i, sorted_dictionary in zip(exclusions, ex_dicts):
    print(f"\n Processing exclusion >= {i}...")

    # Get top 100 memorized and 100 random examples
    hundo_parallel_mem, hundo_parallel_rand, indices_mem, indices_rand = ParallelCorpus(
        range_val=100, sorted_dictionary=sorted_dictionary
    )

    # Detect hallucinations in both sets
    H_mem = algorithm2_from_predictions(
        model_preds=model_preds,
        perturbed_preds=perturbed_preds_full_model, # where do they come from?
        parallel_corpus=hundo_parallel_mem,
        indices=indices_mem,
        T=T,
        model_id=0
    )

    H_rand = algorithm2_from_predictions(
        model_preds=model_preds,
        perturbed_preds=perturbed_preds_full_model, # where do they come from? Line_map?
        parallel_corpus=hundo_parallel_rand,
        indices=indices_rand,
        T=T,
        model_id=0
    )

    # Count unique hallucinations
    ex_mem.append(len(Unique(H_mem)))
    ex_rand.append(len(Unique(H_rand)))

  print(f"    Mem: {ex_mem} | Rand: {ex_rand}")

  return ex_mem, ex_rand

ex_dicts_chrf = [dictionary_chrf1, dictionary_chrf2, dictionary_chrf3, dictionary_chrf4]
ex_dicts_bleu = [dictionary_bleu1, dictionary_bleu2, dictionary_bleu3, dictionary_bleu4]
ex_dicts_acc = [dictionary_acc1, dictionary_acc2, dictionary_acc3, dictionary_acc4]

ex_mem_chrf, ex_rand_chrf = bottom_figure_data(ex_dicts_chrf)
ex_mem_bleu, ex_rand_bleu = bottom_figure_data(ex_dicts_bleu)
ex_mem_acc, ex_rand_acc = bottom_figure_data(ex_dicts_acc)


# graph for Figure 3 (Bottom) chrF
labels = ['>1', '>2', '>3', '>4']

bar_width = 0.4

x = np.arange(len(ex_rand_chrf))

plt.bar(x - bar_width/2, ex_rand_chrf, width=bar_width, label='Random', color='blue')
plt.bar(x + bar_width/2, ex_mem_chrf, width=bar_width, label='Memorized', color='orange')

plt.xlabel('Number of Exclusions')
plt.xticks(x, labels, rotation=45)
plt.ylabel('Unique Hallucinations')

plt.legend()

plt.title('Hallucinations: Random vs Memorized for chrF Metric')
plt.tight_layout()
plt.savefig("figures/hallucinations_random_vs_memorized_chrf.png", dpi=300)
plt.show()

# graph for Figure 3 (Bottom) bleu
labels = ['>1', '>2', '>3', '>4']

bar_width = 0.4

x = np.arange(len(ex_rand_bleu))


plt.bar(x - bar_width/2, ex_rand_bleu, width=bar_width, label='Random', color='blue')
plt.bar(x + bar_width/2, ex_mem_bleu, width=bar_width, label='Memorized', color='orange')

plt.xlabel('Number of Exclusions')
plt.xticks(x, labels, rotation=45)
plt.ylabel('Unique Hallucinations')

plt.legend()

plt.title('Hallucinations: Random vs Memorized for BLEU Metric')
plt.tight_layout()
plt.savefig("figures/hallucinations_random_vs_memorized_bleu.png", dpi=300)
plt.show()

# graph for Figure 3 (Bottom) acc
labels = ['>1', '>2', '>3', '>4']

bar_width = 0.4

x = np.arange(len(ex_rand_acc))

plt.bar(x - bar_width/2, ex_rand_acc, width=bar_width, label='Random', color='blue')
plt.bar(x + bar_width/2, ex_mem_acc, width=bar_width, label='Memorized', color='orange')

plt.xlabel('Number of Exclusions')
plt.xticks(x, labels, rotation=45)
plt.ylabel('Unique Hallucinations')

plt.legend()

plt.title('Hallucinations: Random vs Memorized for Accuracy Metric')
plt.tight_layout()
plt.savefig("figures/hallucinations_random_vs_memorized_acc.png", dpi=300)
plt.show()
