'''
An entry or sent looks like ...

SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O

Each mini-batch returns the followings:
words: list of input sents. ["The 26-year-old ...", ...]
x: encoded input sents. [N, T]. int64.
is_heads: list of head markers. [[1, 1, 0, ...], [...]]
tags: list of tags.['O O B-MISC ...', '...']
y: encoded tags. [N, T]. int64
seqlens: list of seqlens. [45, 49, 10, 50, ...]
'''
import numpy as np
import torch
from torch.utils import data
import json

from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
# VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
VOCAB = (
'<PAD>', 'O', 'I-Business:Merge-Org', 'B-Justice:Pardon', 'B-Justice:Extradite', 'I-Justice:Charge-Indict', 'B-Conflict:Demonstrate', 'I-Justice:Arrest-Jail', 'I-Justice:Acquit', 'B-Life:Divorce',
'B-Life:Be-Born', 'I-Transaction:Transfer-Ownership', 'I-Personnel:Elect', 'I-Justice:Convict', 'I-Life:Injure', 'B-Justice:Execute', 'I-Business:Start-Org', 'B-Business:Merge-Org',
'B-Justice:Appeal', 'I-Personnel:End-Position', 'B-Justice:Convict', 'I-Life:Die', 'B-Business:Declare-Bankruptcy', 'I-Contact:Phone-Write', 'B-Contact:Phone-Write', 'B-Personnel:Start-Position',
'B-Justice:Fine', 'B-Justice:Sue', 'B-Justice:Release-Parole', 'B-Justice:Arrest-Jail', 'I-Life:Marry', 'B-Justice:Trial-Hearing', 'B-Transaction:Transfer-Money', 'B-Justice:Sentence',
'I-Business:End-Org', 'B-Life:Marry', 'I-Conflict:Demonstrate', 'B-Transaction:Transfer-Ownership', 'I-Contact:Meet', 'I-Justice:Sentence', 'I-Transaction:Transfer-Money', 'B-Business:End-Org',
'B-Personnel:Elect', 'B-Contact:Meet', 'B-Justice:Charge-Indict', 'B-Personnel:End-Position', 'I-Justice:Execute', 'B-Personnel:Nominate', 'B-Justice:Acquit', 'I-Justice:Release-Parole',
'I-Conflict:Attack', 'I-Movement:Transport', 'B-Life:Die', 'I-Life:Be-Born', 'B-Life:Injure', 'I-Personnel:Start-Position', 'B-Conflict:Attack', 'B-Business:Start-Org', 'B-Movement:Transport')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}


class TriggerDataset(data.Dataset):
    def __init__(self, fpath):
        """
        fpath: [train|valid|test].txt
        """
        # entries = open(fpath, 'r').read().strip().split("\n\n")
        # sents, tags_li = [], []  # list of lists
        # for entry in entries:
        #     words = [line.split()[0] for line in entry.splitlines()]
        #     tags = ([line.split()[-1] for line in entry.splitlines()])
        #     sents.append(["[CLS]"] + words + ["[SEP]"])
        #     tags_li.append(["<PAD>"] + tags + ["<PAD>"])

        sents, tags_li = [], []  # list of lists
        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item['words']

                tags = ['O'] * len(words)

                for event_mention in item['golden-event-mentions']:
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        if len(tags) <= i:
                            print('out of range! event:', event_mention, words)
                            continue

                        event_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            tags[i] = 'B-{}'.format(event_type)
                        else:
                            tags[i] = 'I-{}'.format(event_type)

                sents.append(["[CLS]"] + words + ["[SEP]"])
                tags_li.append(["<PAD>"] + tags + ["<PAD>"])

        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]  # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], []  # list of ids
        is_heads = []  # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0] * (len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x) == len(y) == len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)

    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens
