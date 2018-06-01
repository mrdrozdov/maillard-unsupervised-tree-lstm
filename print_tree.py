#!/usr/bin/env python3
"""Pretty-print parse trees (only vanilla cyk networks supported)

Usage:
    ./print_tree.py [--inv-temp T] <model-file>

Options:
    --inv-temp T      Inverse temperature for CYK-RNN
"""

import numpy as np
import dynet as dy
from docopt import docopt
import networks
from graphviz import Digraph
from data.utils import cleanup

def build_subtree(weights, words, tree = None, col = None, row = None):
    if tree is None:
        node_attr = { "shape": "plaintext", "colorscheme": "spectral10" }
        graph_attr = { "nodesep": "0.15", "ranksep": "0.15", "ordering": "out" }
        edge_attr = { "arrowhead": "none" }
        tree = Digraph(format = "png", node_attr = node_attr, graph_attr = graph_attr, edge_attr = edge_attr)
        s = Digraph("subgraph", node_attr = node_attr, graph_attr = graph_attr, edge_attr = edge_attr)
        for i in range(len(words)):
            s.node(str((i, 0)), words[i])
        s.graph_attr.update(rank="max")
        tree.subgraph(s)
    if col is None or row is None:
        col = len(weights) - 1
        row = len(weights) - 1

    if row == 0:
        return tree

    my_weights = weights[col][row].npvalue()
    index, max_weight = max(enumerate(my_weights), key=lambda x: x[1])

    avg = 100.0 / float(len(my_weights))
    if avg > 99.0:
        color = "10"
    else:
        q = (10.0 * avg - 100.0) / (avg - 100.0)
        m = - 9.0 / (avg - 100.0)
        color = str(int(round(100.0 * max_weight * m + q)))
    tree.node(str((col, row)), str(int(round(100.0*max_weight))) + "% /" + str(len(my_weights)), fontcolor=color)

    # left
    tree = build_subtree(weights, words, tree, col-row+index, index)
    tree.edge(str((col, row)), str((col-row+index, index)))
    # right
    tree = build_subtree(weights, words, tree, col, row-1-index)
    tree.edge(str((col, row)), str((col, row-1-index)))

    return tree

args = docopt(__doc__)

# load the embeddings and vocab
embeddings = np.load("data/snli2/input_embeddings.npy")
vocab = []
with open("data/snli2/input_vocab.txt") as fin:
    for line in fin:
        vocab.append(line.strip().split("\t")[0])
bacov = {n: i for i, n in enumerate(vocab)}
dim = embeddings.shape[1]

# initialise and restore the model
model = dy.Model()
net = networks.CYK(
    model,
    embeddings,
    update_embeddings = False,
    hidden_dim = 100,
)
model.load(args["<model-file>"])
net.inv_temp = float(args["--inv-temp"])

while True:
    try:
        words = list(map(lambda x: cleanup(x), input("Sentence: ").split(" ")))
        indices = [bacov[w] for w in words if w in bacov]
        n = len(indices)
        _, weights = net(np.reshape(indices, (n, 1)), )
        tree = build_subtree(weights, words)
        tree.render("/home/jm864/public_html/graphviz.gv", view = False)
    except (EOFError, ValueError):
        break
