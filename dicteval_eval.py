#!/usr/bin/env python3
"""Dicteval evaluation script

Usage:
    ./dicteval_eval.py [--inv-temp T] (cyk | ltr-cyk | rtl-cyk | rand-cyk | lstm | tree-lstm | bow) <model-file>

Options:
    -h --help            Show this screen
    --inv-temp T         Inverse temperature for CYK-RNN
"""

import dynet as dy
import numpy as np
import pickle
from docopt import docopt
import networks

# Evaluates a model on the given dictionary embedding task.
# Returns median rank, accuracy@10, accuracy@100.
def eval_dict_dataset(dataset, net, shortlist, proj, parsed):
    ranks = []
    num_batches = len(dataset)
    if parsed:
        dim = dataset[0][0].shape[0]
        batch_size = 1
    else:
        dim, batch_size = dataset[0][0].shape
    for batch_num, data in enumerate(dataset):
        if parsed:
            words, definitions, _ = data
        else:
            words, definitions = data
        words = np.reshape(np.transpose(words), (batch_size, dim))
        dy.renew_cg()
        P = dy.parameter(proj)
        if parsed:
            outputs = net.do_parse_tree(definitions)
        else:
            outputs, _ = net(definitions)
        outputs = P * outputs
        normalised_outputs = outputs * dy.cdiv(dy.inputTensor([1]), dy.sqrt(dy.squared_norm(outputs)))
        normalised_outputs = np.reshape(np.transpose(normalised_outputs.npvalue()), (batch_size, dim))
        for output, word in zip(normalised_outputs, words):
            target_similarity = np.dot(word, output)
            similarities = np.dot(shortlist, output)
            rank = (similarities > target_similarity).sum()
            ranks.append(rank)
    total = len(ranks)
    accuracy10 = float(sum(int(r <= 10) for r in ranks))/total
    accuracy100 = float(sum(int(r <= 100) for r in ranks))/total
    return np.median(ranks), accuracy10, accuracy100

if __name__ == "__main__":
    args = docopt(__doc__)

    model = dy.Model()

    input_embeddings = np.load("data/dicteval/input_embeddings.reduced.npy")
    hidden_dim = 256
    parsed = False

    if args["cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = hidden_dim,
        )
    elif args["ltr-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = hidden_dim,
            order = 1,
        )
    elif args["rtl-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = hidden_dim,
            order = 2,
        )
    elif args["rand-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = hidden_dim,
            order = 3,
        )
    elif args["lstm"]:
        net = networks.LSTM(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = hidden_dim,
        )
    elif args["bow"]:
        net = networks.BOW(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = hidden_dim,
        )
    elif args["tree-lstm"]:
        input_embeddings = np.load("data/dicteval/input_embeddings_parsed.reduced.npy")
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = hidden_dim,
        )
        parsed = True

    if parsed:
        with open("data/dicteval/test_seen_parsed.pkl", "rb") as fin:
            test_seen = pickle.load(fin)
        with open("data/dicteval/test_unseen_parsed.pkl", "rb") as fin:
            test_unseen = pickle.load(fin)
        with open("data/dicteval/test_concepts_parsed.pkl", "rb") as fin:
            test_concepts = pickle.load(fin)
        output_dim = test_seen[0][0].shape[0]
    else:
        with open("data/dicteval/test_seen.pkl", "rb") as fin:
            test_seen = pickle.load(fin)
        with open("data/dicteval/test_unseen.pkl", "rb") as fin:
            test_unseen = pickle.load(fin)
        with open("data/dicteval/test_concepts.pkl", "rb") as fin:
            test_concepts = pickle.load(fin)
        output_dim, _ = test_seen[0][0].shape

    proj = model.add_parameters((output_dim, net.hidden_dim)) 
    shortlist = np.load("data/dicteval/shortlist.npy")
    model.load(args["<model-file>"])
    if args["--inv-temp"] is not None:
        net.inv_temp = float(args["--inv-temp"])

    print("    Seen: "+str(eval_dict_dataset(test_seen, net, shortlist, proj, parsed)))
    print("  Unseen: "+str(eval_dict_dataset(test_unseen, net, shortlist, proj, parsed)))
    print("Concepts: "+str(eval_dict_dataset(test_concepts, net, shortlist, proj, parsed)))
