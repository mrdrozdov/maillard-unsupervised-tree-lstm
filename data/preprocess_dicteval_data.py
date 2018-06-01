#!/usr/bin/env python3
"""
* Builds a vocabulary of input words, and turns all definitions into lists of indices.
* Groups training data points by length of the definition to enable easy batching.
* Performs PCAs on the pretrained embeddings to obtain smaller ones, used in the experiments that also update the embeddings.
"""

import random
import pickle
import numpy as np
from sklearn.decomposition import PCA
from utils import cleanup, OTFVocab, tokenise_and_cleanup_sentence

######################
# Tunable parameters #
######################

max_length = 32  # skip *training* data points with longer definitions
input_dim = 256  # dimension of the reduced input embeddings, for the experiments that use them
batch_size = 16
output_dir = "dicteval/"

# pre-trained embeddings from <http://www.cl.cam.ac.uk/~fh295/dicteval.html>
# (somehow they don't even have full coverage of the test data??)
pre_vocab_path = "/local/filespace/jm864/vectors/felix/vocab.txt"
pre_embs_path = "/local/filespace/jm864/vectors/felix/embeddings.npy"

# data from <http://www.cl.cam.ac.uk/~fh295/dicteval.html>
# (sadly tokenisation is all messed up in this dataset...)
training_data_path = "/local/filespace/jm864/datasets/dicteval/training_data.pkl"
test_data_prefix = "/local/filespace/jm864/datasets/dicteval/"
shortlist_path = "/local/filespace/jm864/datasets/dicteval/shortlist.txt"


#####################
# Load the raw data #
#####################

pre_vocab = []
with open(pre_vocab_path) as fin:
    for line in fin:
        pre_vocab.append(cleanup(line.strip()))
pre_bacov = {n: i for i, n in enumerate(pre_vocab)}  # reverse vocab :)
pre_embs = np.load(pre_embs_path)
pre = lambda x: pre_embs[pre_bacov[x]]
output_dim = pre_embs.shape[1]
print("Loaded pretrained embeddings")

with open(training_data_path, "rb") as fin:
    words, definitions = pickle.load(fin)
print("Loaded training data")

#################
# Training data #
#################

# a vocabulary that is built on-the-fly
input_vocab = OTFVocab()

# bucket the data
buckets = []
for l in range(max_length+1):
    buckets.append([])
for word, definition in zip(words, definitions):
    word = cleanup(word)
    definition = [cleanup(token) for token in definition]
    if len(definition) <= max_length and len(definition) > 0 and word in pre_bacov:
        buckets[len(definition)].append((pre_bacov[word], [input_vocab[token] for token in definition]))

# shuffle the buckets
for bucket in buckets:
    random.shuffle(bucket)

# shuffle the lengths
definition_lengths = []
for definition_length in range(1, max_length+1):
    for _ in range(len(buckets[definition_length]) // batch_size):
        definition_lengths.append(definition_length)
random.shuffle(definition_lengths)
num_batches = len(definition_lengths)
print("Built the training buckets")

# create the training data
training_data = []
for batch_num, definition_length in enumerate(definition_lengths):
    words = []
    definitions = []
    for i in range(0, batch_size):
        word_id, definition_ids = buckets[definition_length].pop()
        words.append(pre_embs[word_id] / np.linalg.norm(pre_embs[word_id]))
        definitions.append(definition_ids)
    words = np.transpose(np.array(words, dtype=np.float32))
    definitions = np.transpose(np.array(definitions, dtype=np.int32))
    training_data.append((words, definitions))
print("Processed "+str(len(definition_lengths))+" batches of size "+str(batch_size))

with open(output_dir+"training.pkl", "wb") as fout:
    pickle.dump(training_data, fout, pickle.HIGHEST_PROTOCOL)

###################
# Evaluation data #
###################

def build_eval(filename):
    data = []
    skipped = 0
    total = 0
    with open(filename, "r") as fin:
        for line in fin:
            total += 1
            word, definition = line.strip().split("\t")
            word = cleanup(word)
            definition = [cleanup(token) for token in tokenise_and_cleanup_sentence(definition)]
            if word not in pre_bacov:
                skipped += 1
                continue
            word = pre_embs[pre_bacov[word]] / np.linalg.norm(pre_embs[pre_bacov[word]])
            definition = [input_vocab.bacov[token] for token in definition if token in input_vocab.bacov]
            if len(definition) < 1:
                skipped += 1
                continue
            # the trailing singleton dimensions below indicate a batch size of 1
            word = np.reshape(np.array(word, dtype=np.float32), (output_dim, 1))
            definition = np.reshape(np.array(definition, dtype=np.int32), (len(definition), 1))
            data.append((word, definition))
    print(filename+": skipped "+str(skipped)+" examples out of "+str(total)+".")
    return data

with open(output_dir+"test_seen.pkl", "wb") as fout:
    data = build_eval(test_data_prefix + "WN_seen_correct.txt")
    pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
with open(output_dir+"test_unseen.pkl", "wb") as fout:
    data = build_eval(test_data_prefix + "WN_unseen_correct.txt")
    pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
with open(output_dir+"test_concepts.pkl", "wb") as fout:
    data = build_eval(test_data_prefix + "concept_descriptions.txt")
    pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)

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
np.save(output_dir+"input_embeddings.full.npy", embeddings)

pca = PCA(n_components = input_dim)
np.save(output_dir+"input_embeddings.reduced.npy", np.array(pca.fit_transform(embeddings), dtype=np.float32))
print("Performed PCA on pre-trained embeddings to obtain reduced input embeddings")

#############
# Shortlist #
#############

shortlist = []
skipped = 0
total = 0
with open(shortlist_path) as f:
    for line in f:
        total += 1
        word = cleanup(line.strip())
        if word in pre_bacov:
            shortlist.append(pre_embs[pre_bacov[word]] / np.linalg.norm(pre_embs[pre_bacov[word]]))
        else:
            skipped += 1
shortlist = np.array(shortlist, dtype=np.float32)
print("Skipped "+str(skipped)+" shortlist words out of "+str(total))
np.save(output_dir+"shortlist.npy", shortlist)
