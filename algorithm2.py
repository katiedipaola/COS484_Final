import numpy as np
import random

def countTopWords(words, k):
    diction = {}
    for i in range(len(words)):
      if words[i] in diction:
        diction[words[i]] += 1
      else:
        diction[words[i]] = 1

    listed = []
    diction2 = {}
    sortkeys = list(diction.keys())
    sortvals = list(diction.values())
    sortedval = np.argsort(sortvals)
    flipsort = sortedval[::-1]
    for i in range(len(flipsort)):
      diction2[sortkeys[flipsort[i]]] = sortvals[flipsort[i]]
    for key, value in diction2.items():
      listed.append((key, value))
    return listed[0:k]

# Load the training data from the file 'train.de'
with open('fairseq/iwslt14.tokenized.de-en/train.de', 'r') as f:
    train_de = f.readlines()
# Split the data into words
train_de_words = []
for line in train_de:
    train_de_words.extend(line.split())

top100 = countTopWords(train_de_words, k=100)

T = random.sample(top100, 30)

print(T)

import json

def sort_mems(file_name):
  with open(file_name, "r") as f:
    memorization_scores = json.load(f)
    print(len(memorization_scores))

  def get_value(item):
      return item[1]

  sorted_values = sorted(memorization_scores.items(), key=get_value, reverse=True)
  sorted_dictionary = dict(sorted_values)

  return sorted_dictionary

with open('fairseq/iwslt14.tokenized.de-en/train.en', 'r') as f:
    train_en = f.readlines()

dictionary_acc = sort_mems("alg1_output/memorization_scores_acc_excl2.json")
dictionary_bleu = sort_mems("alg1_output/memorization_scores_bleu_excl2.json")
dictionary_chrf = sort_mems("alg1_output/memorization_scores_chrf_excl2.json")


def ParallelCorpus(range_val, sorted_dictionary):
    # map the top `range_val` from the sorted_dictionary to the sentence pairs
    top_hundo_de = []
    top_hundo_en = []
    indices_mem = []

    top_items = list(sorted_dictionary.items())[:range_val]
    for key, line_num in top_items:
        key = int(key)
        top_hundo_de.append(train_de[key])
        top_hundo_en.append(train_en[key])
        indices_mem.append(key)

    parallel_mem = list(zip(top_hundo_de, top_hundo_en))

    # from rest of sorted_dictionary, random sample 100 indices and grab those pairs
    sen_left = [i for i in range(len(train_de)) if i not in indices_mem]
    if range_val > len(sen_left):
        raise ValueError("Not enough sentences left to sample")
    sampled = random.sample(sen_left, range_val)
    sampled_sen_de = [train_de[i] for i in sampled]
    sampled_sen_en = [train_en[i] for i in sampled]

    parallel_rand = list(zip(sampled_sen_de, sampled_sen_en))

    return parallel_mem, parallel_rand, indices_mem, sampled

hundo_parallel_mem_acc, hundo_parallel_rand_acc, indices_mem_acc, indices_rand_acc = ParallelCorpus(100, dictionary_acc)
hundo_parallel_mem_acc = list(hundo_parallel_mem_acc)
hundo_parallel_rand_acc = list(hundo_parallel_rand_acc)

hundo_parallel_mem_bleu, hundo_parallel_rand_bleu, indices_mem_bleu, indices_rand_bleu = ParallelCorpus(100, dictionary_bleu)
hundo_parallel_mem_bleu = list(hundo_parallel_mem_bleu)
hundo_parallel_rand_bleu = list(hundo_parallel_rand_bleu)

hundo_parallel_mem_chrf, hundo_parallel_rand_chrf, indices_mem_chrf, indices_rand_chrf = ParallelCorpus(100, dictionary_chrf)
hundo_parallel_mem_chrf = list(hundo_parallel_mem_chrf)
hundo_parallel_rand_chrf = list(hundo_parallel_rand_chrf)


from nltk.translate.bleu_score import sentence_bleu

def adjusted_bleu(generated, actual):
  gen_tokens = generated.strip().split()
  actual_tokens = actual.strip().split()
  weights = (1, 0.8, 0, 0)
  score = sentence_bleu([actual_tokens], gen_tokens, weights=weights)
  return score

def write_perturbed_inputs(parallel_corpus, indices, T, output_path):
    """
    Create a .de file where each line is a perturbed version of a top sentence.
    Also save a line map so you know which (index, token) each line corresponds to.

    Args:
        parallel_corpus: list of (src, tgt) pairs
        indices: list of global dataset indices for those pairs
        T: list of perturbation tokens
        output_path: file to save perturbed inputs to

    Returns:
        line_map: list of (idx, token) corresponding to each line
    """
    line_map = []

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (src, _) in enumerate(parallel_corpus):
            idx = indices[i]
            for t in T:
                perturbed = f"{t} {src.strip()}"
                f.write(perturbed + "\n")
                line_map.append((idx, t))

    return line_map

perturbed_input_path = "content/perturbed_inputs.de"
line_map = write_perturbed_inputs(hundo_parallel_mem_chrf, indices_mem_chrf, T, perturbed_input_path)

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

#print("perturbed_preds")
#print(perturbed_preds_full_model)

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
    print("adjusted bleus")
    
    #print(model_preds)
    for i, (x, y) in enumerate(parallel_corpus):
        #print("here")
        #idx = indices[i]
        idx = str(indices[i])
        #print(idx)

        if idx not in model_preds:
            print("not in model_preds")
            continue

        y_prime = model_preds[idx]
        #print("print bleu:")
        #print(adjusted_bleu(y_prime, y.strip()))
        if adjusted_bleu(y_prime, y.strip()) > 0.09:
            #print("here2")
            for t in T:
                perturbed_key = (int(idx), t)
                #print(perturbed_key)
                if perturbed_key not in perturbed_preds:
                    #print("here3")
                    continue
                print("here3.5")

                y_tilde = perturbed_preds[perturbed_key]
                if adjusted_bleu(y_tilde, y_prime) < 0.01:
                    print("here4")
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

with open("alg1_output/model_preds2.json") as f:
    list_of_dicts = json.load(f)

model_preds = {}
for d in list_of_dicts:
    model_preds.update(d)


H_mem_chrf = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_mem_chrf,
    indices=indices_mem_chrf,
    T=T,
    model_id=0
)
print("H_mem_chrf")
print(H_mem_chrf)

H_rand_chrf = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_rand_chrf,
    indices=indices_rand_chrf,
    T=T,
    model_id=0
)
print("H_rand_chrf")
print(H_rand_chrf)

H_mem_bleu = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_mem_bleu,
    indices=indices_mem_bleu,
    T=T,
    model_id=0
)
print("H_mem_bleu")
print(H_mem_bleu)

H_rand_bleu = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_rand_bleu,
    indices=indices_rand_bleu,
    T=T,
    model_id=0
)
print("H_rand_bleu")
print(H_rand_bleu)

H_mem_acc = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_mem_acc,
    indices=indices_mem_acc,
    T=T,
    model_id=0
)
print("H_mem_acc")
print(H_mem_acc)

H_rand_acc = algorithm2_from_predictions(
    model_preds=model_preds,
    perturbed_preds=perturbed_preds_full_model,
    parallel_corpus=hundo_parallel_rand_acc,
    indices=indices_rand_acc,
    T=T,
    model_id=0
)
print("H_rand_acc")
print(H_rand_acc)

#def Unique(H):
#  non_repeat = []
#  np.sort(H)
#  for i in range(len(H)):
#    if i == 0:
#      non_repeat.append(H[i])
#    elif H[i] != H[i-1]:
#      non_repeat.append(H[i])
#  return non_repeat

def Unique(H):
    seen = set()
    non_repeat = []
    for d in H:
        key = json.dumps(d, sort_keys=True)  # makes dict hashable
        if key not in seen:
            seen.add(key)
            non_repeat.append(d)
    return non_repeat

unique_H_mem_chrf = Unique(H_mem_chrf)
unique_H_rand_chrf = Unique(H_rand_chrf)

unique_H_mem_bleu = Unique(H_mem_bleu)
unique_H_rand_bleu = Unique(H_rand_bleu)

unique_H_mem_acc = Unique(H_mem_acc)
unique_H_rand_acc = Unique(H_rand_acc)


# with BLEU scores onwards:
# data collection for Figure 3 (Top)
def topfiguredata(dictionary):
  sorted_dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
  sorted_dictionary_split = np.array_split(sorted_dictionary, 10)
  print(sorted_dictionary)
  total_mems = []
  unique_mems = []
  for i in range(10):
    subdict = dict(sorted_dictionary_split[i])  # convert chunk back to dict
    parallel_mem, _, indices, _ = ParallelCorpus(range_val=len(subdict), sorted_dictionary=subdict)
    H_mem = algorithm2_from_predictions(model_preds=model_preds,
      perturbed_preds=perturbed_preds_full_model,
      parallel_corpus= parallel_mem,
      indices=indices,
      T=T,
      model_id=0)
    H_mem_len = len(H_mem)
    unique_H_mem = Unique(H_mem)
    unique_H_mem_len = len(unique_H_mem)
    total_mems.append(H_mem_len)
    unique_mems.append(unique_H_mem_len)
  return total_mems, unique_mems

total_mems_acc, unique_mems_acc = topfiguredata(dictionary_acc)
total_mems_bleu, unique_mems_bleu = topfiguredata(dictionary_bleu)
total_mems_chrf, unique_mems_chrf = topfiguredata(dictionary_chrf)

# graph for Figure 3 (Top) acc
import matplotlib.pyplot as plt
import numpy as np

labels = ['Top', '>0.9', '>0.8', '>0.7', '>0.6', '>0.5', '>0.4', '>0.3', '>0.2', '>0.1', '>0']

bar_width = 0.5

pos_unique = np.arange(len(unique_mems_acc))
pos_total = [x + bar_width for x in pos_unique]

plt.bar(pos_unique, unique_mems_acc, width=bar_width, label='Unique HP', color='blue')
plt.bar(pos_total, total_mems_acc, width=bar_width, label='Total HP', color='orange')

plt.xlabel('Memorization Values')
plt.ylabel('Hallucinations')

plt.legend()

plt.title('Hallucinations vs Memorization Values for Accuracy Metric')
plt.tight_layout()
plt.savefig("figures/hallucinations_vs_memorization_acc.png", dpi=300)
plt.show()

# graph for Figure 3 (Top) bleu
import matplotlib.pyplot as plt
import numpy as np

labels = ['Top', '>0.9', '>0.8', '>0.7', '>0.6', '>0.5', '>0.4', '>0.3', '>0.2', '>0.1', '>0']

bar_width = 0.5

pos_unique = np.arange(len(unique_mems_bleu))
pos_total = [x + bar_width for x in pos_unique]

plt.bar(pos_unique, unique_mems_bleu, width=bar_width, label='Unique HP', color='blue')
plt.bar(pos_total, total_mems_bleu, width=bar_width, label='Total HP', color='orange')

plt.xlabel('Memorization Values')
plt.ylabel('Hallucinations')

plt.legend()

plt.title('Hallucinations vs Memorization Values for BLEU Metric')
plt.tight_layout()
plt.savefig("figures/hallucinations_vs_memorization_bleu.png", dpi=300)
plt.show()

# graph for Figure 3 (Top) chrf
labels = ['Top', '>0.9', '>0.8', '>0.7', '>0.6', '>0.5', '>0.4', '>0.3', '>0.2', '>0.1', '>0']

bar_width = 0.5

pos_unique = np.arange(len(unique_mems_chrf))
pos_total = [x + bar_width for x in pos_unique]

plt.bar(pos_unique, unique_mems_chrf, width=bar_width, label='Unique HP', color='blue')
plt.bar(pos_total, total_mems_chrf, width=bar_width, label='Total HP', color='orange')

plt.xlabel('Memorization Values')
plt.ylabel('Hallucinations')

plt.legend()

plt.title('Hallucinations vs Memorization Values for chrF Metric')
plt.tight_layout()
plt.savefig("figures/hallucinations_vs_memorization_chrf.png", dpi=300)
plt.show()



dictionary_bleu1 = sort_mems("alg1_output/memorization_scores_bleu_excl1.json")
dictionary_bleu2 = sort_mems("alg1_output/memorization_scores_bleu_excl2.json")
dictionary_bleu3 = sort_mems("alg1_output/memorization_scores_bleu_excl3.json")
dictionary_bleu4 = sort_mems("alg1_output/memorization_scores_bleu_excl4.json")

dictionary_chrf1 = sort_mems("alg1_output/memorization_scores_chrf_excl1.json")
dictionary_chrf2 = sort_mems("alg1_output/memorization_scores_chrf_excl2.json")
dictionary_chrf3 = sort_mems("alg1_output/memorization_scores_chrf_excl3.json")
dictionary_chrf4 = sort_mems("alg1_output/memorization_scores_chrf_excl4.json")

dictionary_acc1 = sort_mems("alg1_output/memorization_scores_acc_excl1.json")
dictionary_acc2 = sort_mems("alg1_output/memorization_scores_acc_excl2.json")
dictionary_acc3 = sort_mems("alg1_output/memorization_scores_acc_excl3.json")
dictionary_acc4 = sort_mems("alg1_output/memorization_scores_acc_excl4.json")

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

bar_width = 0.5

pos_rand = np.arange(len(ex_rand_chrf))
pos_mem = [x + bar_width for x in pos_rand]

plt.bar(pos_rand, ex_rand_chrf, width=bar_width, label='Random', color='blue')
plt.bar(pos_mem, ex_mem_chrf, width=bar_width, label='Memorized', color='orange')

plt.xlabel('Number of Exclusions')
plt.ylabel('Unique Hallucinations')

plt.legend()

plt.title('Hallucinations: Random vs Memorized for chrF Metric')
plt.tight_layout()
plt.savefig("figures/hallucinations_random_vs_memorized_chrf.png", dpi=300)
plt.show()

# graph for Figure 3 (Bottom) bleu
labels = ['>1', '>2', '>3', '>4']

bar_width = 0.5

pos_rand = np.arange(len(ex_rand_bleu))
pos_mem = [x + bar_width for x in pos_rand]

plt.bar(pos_rand, ex_rand_bleu, width=bar_width, label='Random', color='blue')
plt.bar(pos_mem, ex_mem_bleu, width=bar_width, label='Memorized', color='orange')

plt.xlabel('Number of Exclusions')
plt.ylabel('Unique Hallucinations')

plt.legend()

plt.title('Hallucinations: Random vs Memorized for BLEU Metric')
plt.tight_layout()
plt.savefig("figures/hallucinations_random_vs_memorized_bleu.png", dpi=300)
plt.show()

# graph for Figure 3 (Bottom) acc
labels = ['>1', '>2', '>3', '>4']

bar_width = 0.5

pos_rand = np.arange(len(ex_rand_acc))
pos_mem = [x + bar_width for x in pos_rand]

plt.bar(pos_rand, ex_rand_acc, width=bar_width, label='Random', color='blue')
plt.bar(pos_mem, ex_mem_acc, width=bar_width, label='Memorized', color='orange')

plt.xlabel('Number of Exclusions')
plt.ylabel('Unique Hallucinations')

plt.legend()

plt.title('Hallucinations: Random vs Memorized for Accuracy Metric')
plt.tight_layout()
plt.savefig("figures/hallucinations_random_vs_memorized_acc.png", dpi=300)
plt.show()
