#!/usr/bin/env python3
"""
* Calls CoreNLP parser to produce a parsed version of the dictionary data.
* Saves a separate set of embeddings for parsed data, unlike the SNLI script
"""
import random
import pickle
import numpy as np
from sklearn.decomposition import PCA
from utils import cleanup, OTFVocab, tokenise_and_cleanup_sentence, parse_ptb_sexpr
from pycorenlp import StanfordCoreNLP

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

corenlp = StanfordCoreNLP('http://localhost:9000')
corenlp_properties = { "annotators": "tokenize,ssplit,pos,depparse,parse", "outputFormat": "json" }

# Pretrained embeddings (output space)

pre_vocab = []
with open(pre_vocab_path) as fin:
    for line in fin:
        pre_vocab.append(cleanup(line.strip()))
pre_bacov = {n: i for i, n in enumerate(pre_vocab)}  # reverse vocab :)
pre_embs = np.load(pre_embs_path)

#################
# Training data #
#################

# On-the-fly vocab
input_vocab = OTFVocab()

# Parse data
data = []
for word, definition in zip(words, definitions):
    word = cleanup(word)
    if word not in pre_bacov:
        continue
    l = len(definition)
    if l > max_length:
        continue
    definition = " ".join([cleanup(token) for token in definition])
    corenlp_output = corenlp.annotate(definition, properties = corenlp_properties)
    word = pre_embs[pre_bacov[word]] / np.linalg.norm(pre_embs[pre_bacov[word]])
    if len(corenlp_output["sentences"]) < 1:
        continue
    sexpr = parse_ptb_sexpr(corenlp_output["sentences"][0]["parse"], input_vocab)
    if sexpr is None:
        continue
    data.append((word, sexpr, l))

print("Parsed "+str(len(data))+"/"+str(len(words))+" training examples")

with open(output_dir+"training_parsed.pkl", "wb") as fout:
    pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)

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
            if word not in pre_bacov:
                skipped += 1
                continue
            word = pre_embs[pre_bacov[word]] / np.linalg.norm(pre_embs[pre_bacov[word]])

            definition = [cleanup(token) for token in tokenise_and_cleanup_sentence(definition)]
            definition = [token for token in definition if token in input_vocab.bacov]
            l = len(definition)
            if l < 1:
                skipped += 1
                continue
            corenlp_output = corenlp.annotate(" ".join(definition), properties = corenlp_properties)
            sexpr = parse_ptb_sexpr(corenlp_output["sentences"][0]["parse"], input_vocab.bacov, can_fail = True)
            if sexpr is None:
                skipped += 1
                continue
            data.append((word, sexpr, l))
    print(filename+": skipped "+str(skipped)+" examples out of "+str(total)+".")
    return data

with open(output_dir+"test_seen_parsed.pkl", "wb") as fout:
    data = build_eval(test_data_prefix + "WN_seen_correct.txt")
    pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
with open(output_dir+"test_unseen_parsed.pkl", "wb") as fout:
    data = build_eval(test_data_prefix + "WN_unseen_correct.txt")
    pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
with open(output_dir+"test_concepts_parsed.pkl", "wb") as fout:
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
with open(output_dir+"input_vocab_parsed.txt", "w") as fout:
    for word in input_vocab.vocab:
        if word in pre_bacov:
            embeddings.append(pre_embs[pre_bacov[word]])
        else:
            embeddings.append(unk)
        fout.write(word+"\n")

pca = PCA(n_components = input_dim)
np.save(output_dir+"input_embeddings_parsed.reduced.npy", np.array(pca.fit_transform(embeddings), dtype=np.float32))
print("Performed PCA on pre-trained embeddings to obtain reduced input embeddings")
