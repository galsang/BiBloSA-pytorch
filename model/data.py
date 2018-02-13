from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from nltk import word_tokenize
import numpy as np


class SNLI():
    def __init__(self, args):
        self.TEXT = data.Field(batch_first=True, tokenize=word_tokenize, lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='6B', dim=300))
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_size=args.batch_size,
                                       device=args.gpu)

        self.calculate_block_size(args.batch_size)

    def calculate_block_size(self, B):
        data_lengths = []
        for e in self.train.examples:
            data_lengths.append(len(e.premise))
            data_lengths.append(len(e.hypothesis))

        mean = np.mean(data_lengths)
        std = np.std(data_lengths)

        self.block_size = int((2 * (std * ((2 * np.log(B)) ** (1/2)) + mean)) ** (1/3))
