import sys
import numpy as np
from collections import defaultdict
import random

import torch
from torch.autograd import Variable

import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader(object):
    MAX_VOCAB_SIZE = 50000

    def __init__(self, path, tokens=None, type="word"):
        self.path = path
        self.type = type

        self.datas_train = []
        self.datas_validation = []

        if tokens is None:
            self._load_train_vocab()
            self._load_validation()
        else:
            self.tokens = tokens

        self.token_to_id = dict(zip(
            self.tokens,
            range(len(self.tokens))
        ))


    def _get_tokens(self, line):
        """ Word or char based network
        """

        if self.type == "word":
            return line.split()
        else:
            return list(line)


    def _load_train_vocab(self):
        self.tokens = []

        tokens_count = defaultdict(int)

        with open("{}/train/data.txt".format(self.path), "r") as file:
            print("Load train data...")
            for line in tqdm.tqdm(file):
                tokens = self._get_tokens(line)
                self.datas_train.append(tokens)

                for token in tokens:
                    tokens_count[token] += 1

        np.random.shuffle(self.datas_train)

        print("Tokens calculation...")
        self.tokens = [pair[0] for pair in sorted(
            tokens_count.items(),
            key=lambda x: -x[1]
        )[:DataLoader.MAX_VOCAB_SIZE]]

        self.tokens += ["_PAD_", "_EOS_", "_UNK_"]


    def _load_validation(self):
        with open("{}/validation/data.txt".format(self.path), "r") as file:
            print("Load validation data...")
            for line in tqdm.tqdm(file):
                tokens = self._get_tokens(line)
                self.datas_validation.append(tokens)


    def _get_token_id(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]

        return self.token_to_id["_UNK_"]


    def datas_to_matrix(self, datas, max_len=None):
        max_len = max_len or max(map(len, datas))

        mtx = np.zeros([len(datas), max_len]) + self.token_to_id["_PAD_"]

        for irow in range(len(datas)):
            for itoken, token in enumerate(datas[irow]):
                mtx[irow, itoken]  = self._get_token_id(token)

        return mtx


    def get_random_train_batch(self, batch_size):
        #TODO(dima): process case when batch_size > len(data)
        start_index = random.randint(
            0,
            len(self.datas_train) - batch_size - 1
        )

        end_index = start_index + batch_size

        return Variable(torch.LongTensor(self.datas_to_matrix(
            self.datas_train[start_index:end_index]
        ))).to(device)


    def get_validation_batch_iterator(self, batch_size):
        start_index = 0

        while start_index < len(self.datas_validation):
            end_index = min(
                start_index + batch_size,
                len(self.datas_validation)
            )

            yield Variable(torch.LongTensor(self.datas_to_matrix(
                self.datas_validation[start_index:end_index]
            ))).to(device)

            start_index = end_index


    def get_vocab_size(self):
        return len(self.tokens)
