#!/usr/bin/env python3
"""
* Builds a vocabulary of input words, and turns all definitions into lists of indices.
* Groups training data points by the lengths of both sentences to enable easy batching.
"""

import random
import pickle
import json
import numpy as np
from utils import cleanup, tokenise_and_cleanup_sentence, OTFVocab, parse_ptb_sexpr

# to map labels to integers
label_map = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
}

######################
# Tunable parameters #
######################

batch_size = 16
output_dir = "snli2/"

# pre-trained embeddings from <https://nlp.stanford.edu/projects/glove/>
pre_vocab_path = "/local/filespace/jm864/vectors/glove/vocab.100.txt"
pre_embs_path = "/local/filespace/jm864/vectors/glove/embeddings.100.npy"

# data from <http://nlp.stanford.edu/projects/snli/>
training_data_path = "/local/filespace/jm864/datasets/snli_1.0/snli_1.0_train.jsonl"
test_data_path = "/local/filespace/jm864/datasets/snli_1.0/snli_1.0_test.jsonl"
dev_data_path = "/local/filespace/jm864/datasets/snli_1.0/snli_1.0_dev.jsonl"


##################################
# Load the pretrained embeddings #
##################################

pre_vocab = []
with open(pre_vocab_path) as fin:
    for line in fin:
        pre_vocab.append(cleanup(line.strip()))
pre_bacov = {n: i for i, n in enumerate(pre_vocab)}  # reverse vocab :)
pre_embs = np.load(pre_embs_path)
pre = lambda x: pre_embs[pre_bacov[x]]
output_dim = pre_embs.shape[1]
print("Loaded pretrained embeddings")

#################
# Training data #
#################

# a vocabulary that is built on-the-fly
input_vocab = OTFVocab()

# bucket the data (but the parsed data is saved directly)
parsed_training_data = []
buckets = {}
for l1 in range(1, 200):
    for l2 in range(1, 200):
        buckets[(l1, l2)] = []
skipped = 0
total = 0
with open("/local/filespace/jm864/datasets/snli_1.0/snli_1.0_train.jsonl", "r") as fin:
    for line in fin:
        total += 1
        entry = json.loads(line)
        label = entry["gold_label"]
        if label in label_map:
            label = label_map[label]
        else:
            skipped += 1
            continue
        sentence1 = tokenise_and_cleanup_sentence(entry["sentence1"])
        sentence2 = tokenise_and_cleanup_sentence(entry["sentence2"])
        if len(sentence1) > 200 or len(sentence2) > 200 or len(sentence1) < 1 or len(sentence2) < 1:
            skipped += 1
            continue
        #print((sentence1, sentence2))
        sentence1 = [input_vocab[token] for token in sentence1]
        sentence2 = [input_vocab[token] for token in sentence2]
        buckets[(len(sentence1), len(sentence2))].append((
            label,
            sentence1,
            sentence2,
        ))
        sentence1_sexpr = parse_ptb_sexpr(entry["sentence1_parse"], input_vocab)
        sentence2_sexpr = parse_ptb_sexpr(entry["sentence2_parse"], input_vocab)
        parsed_training_data.append((label, sentence1_sexpr, sentence2_sexpr, len(sentence1), len(sentence2)))
print("Skipped "+str(skipped)+"/"+str(total)+" training examples")

with open(output_dir+"training_parsed.pkl", "wb") as fout:
    pickle.dump(parsed_training_data, fout, pickle.HIGHEST_PROTOCOL)

# shuffle the buckets
for key in buckets.keys():
    random.shuffle(buckets[key])

# shuffle the lengths
sentence_lengths = []
for key in buckets.keys():
    sentence_lengths.extend([key] * (len(buckets[key]) // batch_size))
random.shuffle(sentence_lengths)
num_batches = len(sentence_lengths)
print("Built the training buckets")

# create the training data
training_data = []
for batch_num, pair_lengths in enumerate(sentence_lengths):
    ls = []
    s1 = []
    s2 = []
    for example_num in range(0, batch_size):
        label, sentence1, sentence2, = buckets[pair_lengths].pop()
        ls.append(label)
        s1.append(sentence1)
        s2.append(sentence2)
    ls = np.array(ls, dtype=np.int32)
    s1 = np.transpose(np.array(s1, dtype=np.int32))
    s2 = np.transpose(np.array(s2, dtype=np.int32))
    training_data.append((ls, s1, s2))
print("Processed "+str(len(sentence_lengths))+" batches of size "+str(batch_size))

with open(output_dir+"training.pkl", "wb") as fout:
    pickle.dump(training_data, fout, pickle.HIGHEST_PROTOCOL)

###################
# Evaluation data #
###################

def build_eval(filename):
    buckets = {}
    for l1 in range(1, 200):
        for l2 in range(1, 200):
            buckets[(l1, l2)] = []
    parsed_data = []
    skipped = 0
    total = 0
    with open(filename, "r") as fin:
        for line in fin:
            total += 1
            entry = json.loads(line)
            label = entry["gold_label"]
            if label in label_map:
                label = label_map[label]
            else:
                skipped += 1
                continue
            sentence1 = tokenise_and_cleanup_sentence(entry["sentence1"])
            sentence2 = tokenise_and_cleanup_sentence(entry["sentence2"])
            sentence1 = [input_vocab[token] for token in sentence1]
            sentence2 = [input_vocab[token] for token in sentence2]
            if len(sentence1) < 1 or len(sentence2) < 1:
                skipped += 1
                continue
            sentence1_sexpr = parse_ptb_sexpr(entry["sentence1_parse"], input_vocab)
            sentence2_sexpr = parse_ptb_sexpr(entry["sentence2_parse"], input_vocab)
            if sentence1_sexpr is None or sentence2_sexpr is None:
                skipped +=1
                continue
            buckets[(len(sentence1), len(sentence2))].append((
                label,
                sentence1,
                sentence2,
            ))
            parsed_data.append((label, sentence1_sexpr, sentence2_sexpr))
    data = []
    for bucket in buckets.values():
        ls = []
        s1s = []
        s2s = []
        for l, s1, s2 in bucket:
            ls.append(l)
            s1s.append(s1)
            s2s.append(s2)
            # once we reach batch_size, make a batch
            if len(ls) >= batch_size:
                data.append((
                    np.array(ls, dtype=np.int32),
                    np.transpose(np.array(s1s, dtype=np.int32)),
                    np.transpose(np.array(s2s, dtype=np.int32)),
                ))
                ls = []
                s1s = []
                s2s = []
        # make a batch with the leftovers
        if len(ls) > 0:
            data.append((
                np.array(ls, dtype=np.int32),
                np.transpose(np.array(s1s, dtype=np.int32)),
                np.transpose(np.array(s2s, dtype=np.int32)),
            ))
    print(filename+": skipped "+str(skipped)+" examples out of "+str(total)+".")
    return data, parsed_data

data, parsed_data = build_eval(test_data_path)
with open(output_dir+"test.pkl", "wb") as fout:
    pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
with open(output_dir+"test_parsed.pkl", "wb") as fout:
    pickle.dump(parsed_data, fout, pickle.HIGHEST_PROTOCOL)

data, parsed_data = build_eval(dev_data_path)
with open(output_dir+"dev.pkl", "wb") as fout:
    pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
with open(output_dir+"dev_parsed.pkl", "wb") as fout:
    pickle.dump(parsed_data, fout, pickle.HIGHEST_PROTOCOL)

##############################
# Save pretrained embeddings #
##############################

# calculate UNK by averaging the embeddings we do have in the vocab
unk = np.zeros((output_dim,))
in_vocab = 0
for word in input_vocab.vocab:
    if word in pre_bacov:
        unk += pre_embs[pre_bacov[word]]
        in_vocab += 1
unk = unk / float(in_vocab)
print("Set "+str(len(input_vocab)-in_vocab)+"/"+str(len(input_vocab))+" missing words in the pretrained embeddings to UNK")

# save the mutable embeddings and the vocab
embeddings = []
with open(output_dir+"input_vocab.txt", "w") as fout:
    for word in input_vocab.vocab:
        if word in pre_bacov:
            embeddings.append(pre_embs[pre_bacov[word]])
        else:
            embeddings.append(unk)
        fout.write(word+"\n")
embeddings = np.array(embeddings, dtype=np.float32)
np.save(output_dir+"input_embeddings.npy", embeddings)
