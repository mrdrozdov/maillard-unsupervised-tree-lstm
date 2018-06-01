#!/usr/bin/env python3
"""SNLI training script

Usage:
    ./snli_train.py [options] (cyk | ltr-cyk | rtl-cyk | rand-cyk | lstm | tree-lstm | bow) <save-file>

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
    parsed = False,
    restart = None,
):
    print("Maximum sentence length is set to "+str(max_sentence_length))
    # load pre-parsed data
    if parsed:
        with open("data/snli2/training_parsed.pkl", "rb") as fin:
            training_data = pickle.load(fin)
            training_total = len(training_data)
            training_data = [(l, s1, s2) for l, s1, s2, ls1, ls2  in training_data if ls1 <= max_sentence_length and ls1 > 1 and ls2 <= max_sentence_length and ls2 > 1]
        with open("data/snli2/dev_parsed.pkl", "rb") as fin:
            dev_data = pickle.load(fin)
        num_batches = len(training_data)
        print("Training data contains "+str(num_batches) + " batches (originally "+str(training_total)+") of size 1")
    # or load raw data
    else:
        with open("data/snli2/training.pkl", "rb") as fin:
            training_data = pickle.load(fin)
            training_total = len(training_data)
            training_data = [(l, s1, s2) for l, s1, s2 in training_data if len(s1) <= max_sentence_length and len(s1) > 1 and len(s2) <= max_sentence_length and len(s2) > 1]
        with open("data/snli2/dev.pkl", "rb") as fin:
            dev_data = pickle.load(fin)
            dev_data = [(l, s1, s2) for l, s1, s2 in dev_data if len(s1) <= max_sentence_length and len(s1) > 1 and len(s2) <= max_sentence_length and len(s2) > 1]
        num_batches = len(training_data)
        batch_size = len(training_data[0][0])
        print("Training data contains "+str(num_batches) + " batches (originally "+str(training_total)+") of size "+str(batch_size))

    classifier = networks.SNLIClassifier(model, net.hidden_dim)
    trainer = dy.SimpleSGDTrainer(model, e0=0.01)

    # hyperparameters
    report_frequency = 500
    validate_frequency = num_batches // 10
    if parsed:
        report_frequency = 500 * 16

    start_time = time()
    last_validated = None
    last_reported = None
    best_validation = 0
    validations = []
    validation_means = []
    avg_window_size = 5
    patience = 12
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
            dy.renew_cg()
            ls, s1, s2 = data
            if parsed:
                output1 = net.do_parse_tree(s1)
                output2 = net.do_parse_tree(s2)
            else:
                output1, _ = net(s1)
                output2, _ = net(s2)

            predicted_labels = classifier(output1, output2)
            if parsed:
                loss = dy.pickneglogsoftmax(predicted_labels, ls)
            else:
                loss = dy.sum_batches(dy.pickneglogsoftmax_batch(predicted_labels, ls))

            # optimise
            loss.forward()
            loss.backward()
            trainer.update()

            # Evaluate on development data
            if batches_seen % validate_frequency == 0 and last_validated != batches_seen:
                last_validated = batches_seen
                acc = eval_nli_dataset(net, classifier, dev_data, parsed)
                validations.append(acc)
                validation_means.append(np.mean(validations[-avg_window_size:]))
                print("Validation: accuracy "+str(acc)+", moving average "+str(validation_means[-1]))
                if acc >= best_validation:
                    best_validation = acc
                    model.save(model_name)
                    print("(model saved)")
                    frustration = 0

                # Write to log file
                with open(model_name+".log", "a") as flog:
                    prog = batches_seen
                    if parsed:
                        prog = batches_seen / 16
                    flog.write(str(prog)+"\t"+str(acc)+"\n")

                # Decide if it's time to stop
                if len(validation_means) > patience and validation_means[-1] <= np.array(validation_means[:-patience]).max():
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
                    net.inv_temp = (float(epoch) + fraction_done)*100.0 + 1.0 # max(1.0 / pow(2.0, float(epoch) + fraction_done), 0.005)
                print(
                    "Processed "+str(round(fraction_done*100,2))+"% "+
                    "of epoch #"+str(epoch)+
                    " after "+str(round(elapsed_minutes))+" mins"+
                    (", inv. temp. "+str(net.inv_temp) if isinstance(net, networks.CYK) else "")
                )

            batches_seen += 1
        if early_stop:
            break
        epoch += 1
    print("Training "+str(model_name)+" finished.")

# Evaluates a model on the given SNLI dataset.
def eval_nli_dataset(net, classifier, dataset, parsed):
    accurate = 0.0
    total = 0.0
    for l, s1, s2 in dataset:
        dy.renew_cg()
        if parsed:
            output1 = net.do_parse_tree(s1)
            output2 = net.do_parse_tree(s2)
            total += 1.0
            predicted = classifier(output1, output2).tensor_value().argmax().as_numpy()
            if predicted == l:
                accurate += 1.0
        else:
            output1, _ = net(s1)
            output2, _ = net(s2)
            total += len(l)
            predicted = classifier(output1, output2).tensor_value().argmax().as_numpy()
            r = np.sum(np.equal(l, predicted))
            accurate += r

    return accurate / total

if __name__ == "__main__":
    args = docopt(__doc__)

    model = dy.Model()

    update_embeddings = bool(args["--update-embeddings"])
    input_embeddings = np.load("data/snli2/input_embeddings.npy")

    parsed = False
    if args["cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = 100,
        )
    elif args["ltr-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = 100,
            order = 1,
        )
    elif args["rtl-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = 100,
            order = 2,
        )
    elif args["rand-cyk"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = 100,
            order = 3,
        )
    elif args["lstm"]:
        net = networks.LSTM(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = 100,
        )
    elif args["bow"]:
        net = networks.BOW(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = 100,
        )
    elif args["tree-lstm"]:
        net = networks.CYK(
            model,
            input_embeddings,
            update_embeddings = update_embeddings,
            hidden_dim = 100,
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
        parsed = parsed,
        restart = restart,
    )
