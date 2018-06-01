#!/usr/bin/env python3
"""Dicteval training script

Usage:
    ./dicteval_train.py [options] (cyk | ltr-cyk | rtl-cyk | rand-cyk | lstm | tree-lstm | bow) <save-file>

Options:
    -h --help            Show this screen
    --update-embeddings  Update input embeddings
    --max-length M       Skip sentences longer than M words during
                         training [default: 999]
    --restart N          Restart interrupted training from a given epoch

DyNet Options:
    --dynet-gpu-ids IDS     Device ids of the GPUs to be used
    --dynet-weight-decay W  Apply weight decay W
"""

import dynet as dy
import numpy as np
import pickle
from docopt import docopt
from time import time
import networks

# Trains the model in a loop until the early stopping criterion is met.
# Saves whenever the performance on development data improves.
def train_model(
    net,
    model_name,
    max_sentence_length,
    restart = None,
    parsed = False,
):
    shortlist = np.load("data/dicteval/shortlist.npy")
    print("Max sentence length is "+str(max_sentence_length))

    # load training and evaluation data
    if parsed:
        with open("data/dicteval/training_parsed.pkl", "rb") as fin:
            training_data = pickle.load(fin)
            training_total = len(training_data)
            training_data = [(ws, ds) for ws, ds, l in training_data if l <= max_sentence_length and l > 0]
    else:
        with open("data/dicteval/training.pkl", "rb") as fin:
            training_data = pickle.load(fin)
            training_total = len(training_data)
            training_data = [(ws, ds) for ws, ds in training_data if ds.shape[0] <= max_sentence_length and ds.shape[0] > 0]

    # keep some data for validation
    if parsed:
        dev_size = 128 * 16
    else:
        dev_size = 128
    print("Reserving "+str(dev_size)+" batches as development set")
    dev_data = training_data[:dev_size]
    training_data = training_data[dev_size:]

    num_batches = len(training_data)
    if parsed:
        batch_size = 1
        output_dim = training_data[0][0].shape[0]
    else:
        output_dim, batch_size = training_data[0][0].shape

    trainer = dy.SimpleSGDTrainer(model, e0=0.01)

    # projection matrix onto "output" embedding space
    proj = model.add_parameters((output_dim, net.hidden_dim)) 

    print("Training data contains "+str(num_batches) + " batches (originally "+str(training_total-dev_size)+") of size "+str(batch_size))

    # hyperparameters
    report_frequency = 500
    validate_frequency = num_batches // 15
    if parsed:
        report_frequency = 500 * 16

    start_time = time()
    last_validated = None
    last_reported = None
    best_validation = shortlist.shape[0]
    validations = []
    validation_means = []
    avg_window_size = 5
    patience = 10
    frustration = 0
    early_stop = False
    epoch = 0
    batches_seen = 0
    if isinstance(restart, int):
        model.load(model_name)
        epoch = restart
        batches_seen = epoch * num_batches
        print("Restarting interrupted training from epoch "+str(epoch))
    while True:
        print("Start of epoch #"+str(epoch))
        for batch_num, data in enumerate(training_data):
            words, definitions = data
            # Perform the composition
            dy.renew_cg()
            if parsed:
                outputs = net.do_parse_tree(definitions)
            else:
                outputs, _ = net(definitions)

            # Calculate the loss and optimise
            P = dy.parameter(proj)
            if parsed:
                word_embeddings = dy.inputTensor(words, batched=False)
                loss = cosine_similarity(P * outputs, word_embeddings) * (-1.0/batch_size)
            else:
                word_embeddings = dy.inputTensor(words, batched=True)
                loss = dy.sum_batches(cosine_similarity(P * outputs, word_embeddings) * (-1.0/batch_size))
            loss.forward()
            loss.backward()
            trainer.update()

            # Evaluate on development data
            if batches_seen % validate_frequency == 0 and last_validated != batches_seen:
                last_validated = batches_seen
                median, acc_at_10, acc_at_100 = eval_dict_dataset(dev_data, net, shortlist, proj, parsed)
                validations.append(median)
                validation_means.append(np.mean(validations[-avg_window_size:]))
                print("Validation: median "+str(median)+", moving average "+str(validation_means[-1]))

                # If we have a new best, reset frustration
                if median <= best_validation:
                    best_validation = median
                    model.save(model_name)
                    print("(new best)")
                    frustration = 0

                # Write to log file
                with open(model_name+".log", "a") as flog:
                    prog = batches_seen
                    if parsed:
                        prog = batches_seen / 16
                    flog.write(str(prog)+"\t"+str(median)+"\n")

                # Decide if it's time to stop
                if len(validation_means) > patience and validation_means[-1] > np.array(validation_means[:-patience]).min():
                    frustration += 1
                    if frustration > patience:
                        print("Early stop!")
                        early_stop = True
                        break
                else:
                    frustration = 0

            # Report progress
            if batches_seen % report_frequency == 0 and last_reported != batches_seen:
                last_reported = batches_seen
                fraction_done = batch_num / num_batches
                elapsed_minutes = (time() - start_time)/60.0
                # Update temperature
                if isinstance(net, networks.CYK):
                    net.inv_temp = pow(1.75, float(epoch) + fraction_done)
                print(
                    "Processed "+str(round(fraction_done * 100, 2))+"% "+
                    "of epoch #"+str(epoch)+
                    " after "+str(round(elapsed_minutes))+" mins"+
                    (", inv. temp. "+str(net.inv_temp) if isinstance(net, networks.CYK) else "")
                )

            batches_seen += 1
        if early_stop:
            break
        epoch += 1

    print("Training "+str(model_name)+" finished.")

# Cosine similarity. NOTE: the second argument is assumed to be already normalised
# (it's the "output embeddings", which stays constant)
def cosine_similarity(a, b):
    # FIXME do I really need to do this for scalar division? :(
    norm = dy.cdiv(dy.inputTensor([1]), dy.sqrt(dy.squared_norm(a)))
    return dy.dot_product(a, b) * norm 

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

    update_embeddings = bool(args["--update-embeddings"])
    input_embeddings = np.load("data/dicteval/input_embeddings.reduced.npy")
    hidden_dim = 256
    parsed = False

    if args["cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = hidden_dim,
        )
    elif args["ltr-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = hidden_dim,
            order = 1,
        )
    elif args["rtl-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = hidden_dim,
            order = 2,
        )
    elif args["rand-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = hidden_dim,
            order = 3,
        )
    elif args["lstm"]:
        net = networks.LSTM(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = hidden_dim,
        )
    elif args["bow"]:
        net = networks.BOW(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = hidden_dim,
        )
    elif args["tree-lstm"]:
        input_embeddings = np.load("data/dicteval/input_embeddings_parsed.reduced.npy")
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = hidden_dim,
        )
        parsed = True

    if args["--restart"]:
        restart = int(args["--restart"])
    else:
        restart = None

    train_model(
        net,
        model_name = args["<save-file>"],
        max_sentence_length = int(args["--max-length"]),
        restart = restart,
        parsed = parsed
    )
