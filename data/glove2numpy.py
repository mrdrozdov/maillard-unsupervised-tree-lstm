import sys
import numpy as np

VERBOSE = True

def noop(x):
    return x

try:
    from tqdm import tqdm as tqdm
except:
    tqdm = noop

if not VERBOSE:
    tqdm = noop


# np.save(outfile, x)
# np.load(outfile)


def get_vocab_size(glove_in):
    count = 0
    with open(glove_in) as f:
        for line in f:
            count += 1
    return count


def get_embedding_size(glove_in):
    with open(glove_in) as f:
        for line in f:
            parts = line.strip().split(' ')
            # subtract one for vocab word
            length = len(parts) - 1
            break
    return length


def get_vocab_and_embeddings(glove_in, vocab_size=None, embedding_size=None):
    if vocab_size is None:
        vocab_size = get_vocab_size(glove_in)
    if embedding_size is None:
        embedding_size = get_embedding_size(glove_in)

    vocab = [None] * vocab_size
    embeddings = np.empty((vocab_size, embedding_size), dtype=np.float32)

    with open(glove_in) as f:
        for i, line in tqdm(enumerate(f)):
            parts = line.strip().split(' ')
            vocab[i] = parts[0]
            embeddings[i] = [float(x) for x in parts[1:]]

    return vocab, embeddings


def save_vocab(vocab, vocab_out):
    with open(vocab_out, 'w') as f:
        for w in vocab:
            f.write('{}\n'.format(w))


def save_embeddings(embeddings, numpy_out):
    np.save(numpy_out, embeddings)


def convert(glove_in, vocab_out, numpy_out):
    vocab, embeddings = get_vocab_and_embeddings(glove_in)
    save_vocab(vocab, vocab_out)
    save_embeddings(embeddings, numpy_out)


if __name__ == '__main__':
    """
    Usage:
        python glove2numpy.py /path/in/glove.txt /path/out/vocab.txt /path/out/glove.npy

    Example:
        python glove2numpy.py ~/data/glove/glove.6B.100d.txt ~/data/glove/glove.vocab.6B.100d.txt ~/data/glove/glove.embeddings.6B.100d.npy

    """
    glove_in = sys.argv[1]
    vocab_out = sys.argv[2]
    numpy_out = sys.argv[3]

    convert(glove_in, vocab_out, numpy_out)
