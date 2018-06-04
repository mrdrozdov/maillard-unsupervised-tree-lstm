import sys
import json
import pickle

VERBOSE = True

def noop(x, *args, **kwargs):
    return x

try:
    from tqdm import tqdm as tqdm
except:
    tqdm = noop

if not VERBOSE:
    tqdm = noop


LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}


def label2index(label):
    return LABEL_MAP[label]


def read_snli(filename):
    """
    l, s1, s2
    """
    data = []
    with open(filename) as f:
        for line in tqdm(f, desc='read_snli'):
            example = json.loads(line.strip())
            try:
                label = label2index(example['gold_label'])
            except:
                continue
            sent1 = example['sentence1'].split(' ')
            sent2 = example['sentence2'].split(' ')
            data.append((label, sent1, sent2))
    return data


def read_snli_parsed(filename):
    """
    l, s1, s2, ls1, ls2
    """
    data = []
    with open(filename) as f:
        for line in tqdm(f, desc='read_snli_parsed'):
            example = json.loads(line.strip())
            try:
                label = label2index(example['gold_label'])
            except:
                continue
            sent1 = example['sentence1'].split(' ')
            sent2 = example['sentence2'].split(' ')
            data.append((label, sent1, sent2, len(sent1), len(sent2)))
    return data


def save(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    """
    Usage:
        python snli2pickle.py ~/data/snli_1.0/snli_1.0_dev.jsonl \
            ~/data/snli_1.0/snli_1.0_dev.sentence.pkl \
            ~/data/snli_1.0/snli_1.0_dev.parsed.pkl
        python snli2pickle.py ~/data/snli_1.0/snli_1.0_train.jsonl \
            ~/data/snli_1.0/snli_1.0_train.sentence.pkl \
            ~/data/snli_1.0/snli_1.0_train.parsed.pkl

    """
    snli_in = sys.argv[1]
    out = sys.argv[2]
    parsed_out = sys.argv[3]

    save(read_snli(snli_in), out)
    save(read_snli_parsed(snli_in), parsed_out)
