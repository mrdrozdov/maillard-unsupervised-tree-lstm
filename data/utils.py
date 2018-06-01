from nltk.tokenize import TreebankWordTokenizer
import re

t = TreebankWordTokenizer()

t.PUNCTUATION = [
    (re.compile(r'([:,])([^\d])'), r' \1 \2'),
    (re.compile(r'([\\/:,])$'), r' \1 '),
    (re.compile(r'\.\.\.'), r' ... '),
    (re.compile(r'[;@#$%&]'), r' \g<0> '),
    (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
    (re.compile(r'[?!]'), r' \g<0> '),
    (re.compile(r"([^'])' "), r"\1 ' "),
]

# token replacements, to attempt some kind of normalisation
replacements = {
  "-lrb-": "(",
  "-rrb-": ")",
  "-lsb-": "[",
  "-rsb-": "]",
  "-lcb-": "{",
  "-rcb-": "}",
  "``": "\"",
  "“": "\"",
  "''": "\"",
  "”": "\"",
  "`": "'",
  "‘": "'",
  "’": "'",
  "---": "--",
  "t.v.": "tv"
}

def cleanup(token):
    token = token.lower()
    if token in replacements:
        return replacements[token]
    else:
        return token

def tokenise_and_cleanup_sentence(sentence):
    sentence = sentence.replace("T.V.", "TV")
    return list(map(cleanup, t.tokenize(sentence)))

# vocabulary created on-the-fly
class OTFVocab:
    def __init__(self):
        self.vocab = []
        self.bacov = {}  # reverse vocab :)

    def __getitem__(self, word):
        word = cleanup(word)
        if word in self.bacov:
            return self.bacov[word]
        else:
            self.bacov[word] = len(self.vocab)
            self.vocab.append(word)
            return len(self.vocab)-1

    def __contains__(self, key):
        return cleanup(key) in self.bacov

    def __len__(self):
        return len(self.vocab)

# parses PTB-style s-exprs into binarised nested tuples, optionally turning
# words into embedding indices. Nonterminals are ignored.
def parse_ptb_sexpr(sexpr, reverse_vocab=None, can_fail=False):
    result, _ = _parse_ptb_sexpr(sexpr.strip(), 0, reverse_vocab, can_fail)
    if can_fail and result is None:
        return None
    if len(result) == 1:
        return result[0]
    else:
        return result

def _parse_ptb_sexpr(sexpr, position, reverse_vocab, can_fail):
    if sexpr[position] != "(":
        raise ValueError("Missing opening parenthesis")
    i = position + 1

    nonterminal_done = False
    node = []
    while i < len(sexpr):
        if sexpr[i] == " " or sexpr[i] == "\n":
            i += 1
        elif sexpr[i] == "(":
            child, child_end = _parse_ptb_sexpr(sexpr, i, reverse_vocab, can_fail)
            if can_fail and child is None:
                return None, None
            if len(child) == 1:
                node.append(child[0])
            else:
                node.append(child)
            i = child_end + 1
        elif sexpr[i] == ")":
            if len(node) < 3:
                return tuple(node), i
            else:  # binarise! (right-branching)
                right = node.pop()
                left = node.pop()
                binarised_node = (left, right)
                while len(node) > 0:
                    binarised_node = (node.pop(), binarised_node)
                return binarised_node, i
        else:
            string = ""
            while sexpr[i] != "(" and sexpr[i] != ")" and sexpr[i] != " ":
                string += sexpr[i]
                i += 1
            # we do not want to process the nonterminal...
            if not nonterminal_done:
                nonterminal_done = True
            # ...but terminals get cleaned up and added
            else:
                string = cleanup(string)
                if reverse_vocab is not None:
                    if can_fail and string not in reverse_vocab:
                        return None, None
                    node.append(reverse_vocab[string])
                else:
                    node.append(string)

    raise ValueError("Missing closing parenthesis")

