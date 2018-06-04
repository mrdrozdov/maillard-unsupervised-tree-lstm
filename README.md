Note: This is a modified version of the original code for use in reproducing the original experiments and related research.

# Jointly Learning Sentence Embeddings and Syntax with Unsupervised Tree-LSTMs

This is the cleaned-up code for the paper “Jointly Learning Sentence Embeddings and Syntax with Unsupervised Tree-LSTMs”, currently under review and available at <https://arxiv.org/abs/1705.09189>. Please refer to <http://www.maillard.it/> for the latest information about the paper.

The code for preprocessing the datasets is under `data/`. The file `networks.py` contains implementations of the various network architectures. You must select which architecture to use when running the `*_train.py`/`*_eval.py` files:
* `bow`: Bag of Words
* `lstm`: LSTM
* `tree-lstm`: Supervised Tree-LSTM
* `ltr-cyk`: Left-branching Tree-LSTM
* `rtl-cyk`: Right-branching Tree-LSTM
* `rand-cyk`: Random-branching non-projective Tree-LSTM
* `cyk`: Unsupervised Tree-LSTM

## Creating glove vocab/embedding files

Example:

```
python data/glove2numpy.py \
    ~/data/glove/glove.6B.100d.txt \
    ~/data/glove/glove.vocab.6B.100d.txt \
    ~/data/glove/glove.embeddings.6B.100d.npy
```
