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

with open('/content/fairseq/iwslt14.tokenized.de-en/train.en', 'r') as f:
    train_en = f.readlines()

dictionary_acc = sort_mems("memorization_scores_acc.json")
dictionary_bleu = sort_mems("memorization_scores_bleu.json")
dictionary_chrf = sort_mems("memorization_scores_chrf.json")


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

perturbed_input_path = "/content/perturbed_inputs.de"
line_map = write_perturbed_inputs(hundo_parallel_mem_chrf, indices_mem_chrf, T, perturbed_input_path)

