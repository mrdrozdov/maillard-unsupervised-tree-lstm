#!/usr/bin/env python3
"""SNLI evaluation script

Usage:
    ./snli_eval.py [--inv-temp T] (cyk | ltr-cyk | rtl-cyk | rand-cyk | lstm | tree-lstm | bow) <model-file>

Options:
    -h --help            Show this screen
    --inv-temp T         Inverse temperature for CYK-RNN
"""

import dynet as dy
import numpy as np
import pickle
from docopt import docopt
import networks

# Evaluates a model on the given SNLI dataset.
def eval_nli_dataset(net, classifier, dataset, parsed):
    accurate = 0.0
    total = 0.0
    for l, s1, s2 in dataset:
        dy.renew_cg()
        if parsed:
            output1 = net.do_parse_tree(s1)
            output2 = net.do_parse_tree(s2)
        else:
            output1, _ = net(s1)
            output2, _ = net(s2)
        predicted = classifier(output1, output2).tensor_value().argmax().as_numpy()
        r = np.sum(np.equal(l, predicted))
        accurate += r
        total += len(l)

    return accurate / total

if __name__ == "__main__":
    args = docopt(__doc__)

    # load test data
    with open("data/snli/test.pkl", "rb") as fin:
        test = pickle.load(fin)
    input_embeddings = np.load("data/snli/input_embeddings.npy")

    model = dy.Model()

    parsed = False
    if args["cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = 100,
        )
        if args["--inv-temp"] is not None:
            net.inv_temp = float(args["--inv-temp"])
            print("Inverse temperature set to "+str(net.inv_temp))
    elif args["ltr-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = 100,
            order = 1,
        )
    elif args["rtl-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = 100,
            order = 2,
        )
    elif args["rand-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = 100,
            order = 3,
        )
    elif args["lstm"]:
        net = networks.LSTM(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = 100,
        )
    elif args["bow"]:
        net = networks.BOW(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = 100,
        )
    elif args["tree-lstm"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = False,
            hidden_dim = 100,
        )
        parsed = True

    classifier = networks.SNLIClassifier(model, input_embeddings.shape[1])
    model.load(args["<model-file>"])
    print(eval_nli_dataset(net, classifier, test, parsed))
