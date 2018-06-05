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
            yield example['sentence1']
            yield example['sentence2']
    return data


def save(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            f.write('{}\n'.format(line))


if __name__ == '__main__':
    """
    Usage:
        python snli2text.py ~/data/snli_1.0/snli_1.0_dev.jsonl \
            ~/data/snli_1.0/snli_1.0_dev.sentences.txt
        

    """
    snli_in = sys.argv[1]
    out = sys.argv[2]

    save(read_snli(snli_in), out)
