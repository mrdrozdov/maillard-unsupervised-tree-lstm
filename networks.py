import dynet as dy
import numpy as np
from collections import namedtuple

# This is the basic unit the chart is made up of, representing the state of
# the Tree-LSTM at a given node.  The elements of the tuple are hidden state
# and memory cell state.
StateTuple = namedtuple("StateTuple", ("h", "c"))

# CYK-RNN architecture a.k.a Unsupervised Tree-LSTM
class CYK:
    # Initialises the network's parameters and data structures
    def __init__(
        self,
        model,
        input_embeddings,
        update_embeddings = False,
        hidden_dim = 512,
        order = 0,  # 0 = standard, 1 = ltr, 2 = rtl, 3 = random non-projective tree
    ):
        self.order = int(order)

        ## Embeddings ##

        self.input_dim = input_embeddings.shape[1]
        self.hidden_dim = int(hidden_dim)

        self.update_embeddings = bool(update_embeddings)
        self.embeddings = model.add_lookup_parameters(input_embeddings.shape)
        self.embeddings.init_from_array(input_embeddings)

        ## Weighting/gating mechanism ##

        self.w_score = model.add_parameters((self.hidden_dim, ))
        self.inv_temp = 10

        ## Tree-LSTM ##

        self.W = model.add_parameters((5 * self.hidden_dim, self.input_dim))
        self.U = model.add_parameters((5 * self.hidden_dim, 2 * self.hidden_dim))
        self.b = model.add_parameters((5 * self.hidden_dim, ))

        print(
            "Set up CYK-RNN model with:\n"+
            "  order: "+str(self.order)+"\n"+
            "  input_dim: "+str(self.input_dim)+"\n"+
            "  hidden_dim: "+str(self.hidden_dim)+"\n"+
            "  update_embeddings: "+str(self.update_embeddings)
        )

    def tree_lstm(self, L=None, R=None, x=None):
        if not (L is not None and R is not None and x is None) and not (x is not None and L is None and R is None):
            raise ValueError("Invalid combination of arguments to Tree-LSTM: "+str((L, R, x)))

        hid = self.hidden_dim
        W, U, b = dy.parameter(self.W), dy.parameter(self.U), dy.parameter(self.b)

        if L is not None and R is not None:
            preact = b + U * dy.concatenate([L.h, R.h])
            i = dy.logistic(preact[:hid])
            fL = dy.logistic(preact[hid:2*hid] + 1.0)
            fR = dy.logistic(preact[2*hid:3*hid] + 1.0)
            o = dy.logistic(preact[3*hid:4*hid])
            u = dy.tanh(preact[4*hid:])
            c = dy.cmult(fL, L.c) + dy.cmult(fR, R.c) + dy.cmult(i, u)
        else:
            preact = b + W * x
            i = dy.logistic(preact[:hid])
            o = dy.logistic(preact[3*hid:4*hid])
            u = dy.tanh(preact[4*hid:])
            c = dy.cmult(i, u)

        h = dy.cmult(o, dy.tanh(c))
        return StateTuple(h = h, c = c)

    def score_constituent(self, constituent):
        w_score = dy.parameter(self.w_score)
        w_score = dy.cdiv(w_score, dy.sqrt(dy.squared_norm(w_score)))
        h = dy.cdiv(constituent.h, dy.sqrt(dy.squared_norm(constituent.h)))
        return dy.dot_product(w_score, h)*self.inv_temp

    def baseline(self, sentences):
        # LTR / random non-projective
        if self.order == 1 or self.order == 3:
            if self.order == 3:
                np.random.shuffle(sentences)
            vecs = [self.tree_lstm(
                L = None,
                R = None,
                x = dy.lookup_batch(self.embeddings, sentences[i], update=self.update_embeddings),
            ) for i in range(sentences.shape[0])]

            state = vecs[0]
            for i in range(1, sentences.shape[0]):
                state = self.tree_lstm(L = state, R = vecs[i], x = None)

        # RTL
        elif self.order == 2:
            vecs = [self.tree_lstm(
                L = None,
                R = None,
                x = dy.lookup_batch(self.embeddings, sentences[i], update=self.update_embeddings),
            ) for i in range(sentences.shape[0])]
            state = vecs[0]
            for i in range(1, sentences.shape[0]):
                state = self.tree_lstm(L = vecs[i], R = state, x = None)

        else:
            raise ValueError("Invalid composition order "+str(self.order))

        return state.h

    # Runs the network on a single parse tree, returning the sentence embedding
    def do_parse_tree(self, tree):
        return self._do_parse_tree(tree).h

    def _do_parse_tree(self, tree):
        if isinstance(tree, int):
            return self.tree_lstm(
                L = None,
                R = None,
                x = dy.lookup(self.embeddings, tree, update=self.update_embeddings),
            )
        elif isinstance(tree, tuple) and len(tree) == 2:
            return self.tree_lstm(
                L = self._do_parse_tree(tree[0]),
                R = self._do_parse_tree(tree[1]),
            )
        else:
            raise ValueError("Malformed tree: "+str(tree))

    # Runs the network on a batch of input sentences, returns the sentence
    # embeddings and constituent weights.
    #
    # `sentences` must be a numpy array with shape `(sentence_length, batch_size)`
    def __call__(self, sentences, argmax=False):
        # baselines that use deterministic or random composition order
        if self.order != 0:
            return self.baseline(sentences), None

        # initialise the empty chart
        length = sentences.shape[0]
        chart = []
        weights = []
        for col in range(length):
            chart.append([self.tree_lstm(
                L = None,
                R = None,
                x = dy.lookup_batch(self.embeddings, sentences[col], update=self.update_embeddings)
            )])
            weights.append([1.0])
            # initialise the remaining chart cells
            for row in range(1, col+1):
                chart[col].append(None)
                weights[col].append(None)

        # fill up the upper rows
        for col in range(length):
            for row in range(1, col+1):
                # at row k, there are k possible way to form each constituent.
                # we try all of them, and keep the results around together with
                # the corresponding scores
                constituents = []
                scores = []
                for constituent_number in range(0, row):
                    constituents.append(self.tree_lstm(
                        L = chart[col-row+constituent_number][constituent_number],
                        R = chart[col][row-1-constituent_number],
                    ))
                    scores.append(
                        self.score_constituent(constituents[-1])
                    )
                # we softmax the weights, and use them as a weighting mechanism
                # that strongly prefers assigning probability mass to only one
                # possibility
                weights[col][row] = dy.softmax(dy.concatenate(scores))
                chart[col][row] = StateTuple(
                    h = dy.esum([c.h * w for w, c in zip(weights[col][row], constituents)]),
                    c = dy.esum([c.c * w for w, c in zip(weights[col][row], constituents)])
                )

        return chart[length-1][length-1].h, weights

# BOW architecture
class BOW:
    # Initialises the network's parameters
    def __init__(
        self,
        model,
        input_embeddings,
        update_embeddings = False,
        hidden_dim = 512,
    ):
        ## Embeddings ##

        self.input_dim = input_embeddings.shape[1]
        self.hidden_dim = int(hidden_dim)

        self.update_embeddings = bool(update_embeddings)
        self.embeddings = model.add_lookup_parameters(input_embeddings.shape)
        self.embeddings.init_from_array(input_embeddings)

        ## Affine transformation ##
        self.A = model.add_parameters((self.hidden_dim, self.input_dim))
        self.a = model.add_parameters((self.hidden_dim,))

        print(
            "Set up BOW model with:\n"+
            "  input_dim: "+str(self.input_dim)+"\n"+
            "  hidden_dim: "+str(self.hidden_dim)+"\n"+
            "  update_embeddings: "+str(self.update_embeddings)
        )

    # Transforms an input vector
    def transform(self, vec):
        a = dy.parameter(self.a)
        A = dy.parameter(self.A)
        return dy.tanh(dy.affine_transform([a, A, vec]))

    # Runs the network on a batch of input sentences, returns the sentence embeddings.
    #
    # `sentences` must be a numpy array with shape `(sentence_length, batch_size)`
    def __call__(self, sentences):
        length = sentences.shape[0]
        vecs = [self.transform(dy.lookup_batch(self.embeddings, sentences[i], update=self.update_embeddings)) for i in range(length)]
        return dy.esum(vecs) * (1.0/float(length)), None

# LSTM architecture
class LSTM:
    # Initialises the network's parameters
    def __init__(
        self,
        model,
        input_embeddings,
        update_embeddings = False,
        hidden_dim = 512,
    ):
        ## Embeddings ##
        self.input_dim = input_embeddings.shape[1]
        self.hidden_dim = int(hidden_dim)

        self.update_embeddings = bool(update_embeddings)
        self.embeddings = model.add_lookup_parameters(input_embeddings.shape)
        self.embeddings.init_from_array(input_embeddings)

        ## LSTM ##
        self.fw_lstm = dy.VanillaLSTMBuilder(1, self.input_dim, self.hidden_dim, model, ln_lstm = False)

        print(
            "Set up LSTM model with:\n"+
            "  input_dim: "+str(self.input_dim)+"\n"+
            "  hidden_dim: "+str(self.hidden_dim)+"\n"+
            "  update_embeddings: "+str(self.update_embeddings)
        )

    # Runs the network on a batch of input sentences, returns the sentence embeddings.
    #
    # `sentences` must be a numpy array with shape `(sentence_length, batch_size)`
    def __call__(self, sentences):
        length = sentences.shape[0]
        vecs = [dy.lookup_batch(self.embeddings, sentences[i], update=self.update_embeddings) for i in range(length)]

        forward = self.fw_lstm.initial_state().transduce(vecs)[-1]
        return forward, None

# Classifier for natural language inference (Bowman et al., ACL 2016)
class SNLIClassifier:
    def __init__(self, model, dim):
        self.b_nli = model.add_parameters((2*dim,))  # NLI layer bias
        self.W_nli_1 = model.add_parameters((2*dim, dim))  # NLI layer weight for first sentence
        self.W_nli_2 = model.add_parameters((2*dim, dim))  # NLI layer weight for second sentence
        self.W_nli_u = model.add_parameters((2*dim, dim))  # NLI layer weight for squared distance
        self.W_nli_v = model.add_parameters((2*dim, dim))  # NLI layer weight for componentwise product
        self.b_s = model.add_parameters((3,))  # softmax bias
        self.w_s = model.add_parameters((3, 2*dim))  # softmax weight

    # Returns the energy to be passed to a softmax
    def __call__(self, s1, s2):
        b_nli = dy.parameter(self.b_nli)
        W_nli_1 = dy.parameter(self.W_nli_1)
        W_nli_2 = dy.parameter(self.W_nli_2)
        W_nli_u = dy.parameter(self.W_nli_u)
        W_nli_v = dy.parameter(self.W_nli_v)
        u = dy.square(s1 - s2)
        v = dy.cmult(s1, s2)
        relu = dy.rectify(dy.affine_transform([b_nli, W_nli_1, s1, W_nli_2, s2, W_nli_u, u, W_nli_v, v]))

        b_s = dy.parameter(self.b_s)
        w_s = dy.parameter(self.w_s)
        return dy.affine_transform([b_s, w_s, relu])

