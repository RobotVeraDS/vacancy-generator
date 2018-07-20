import sys
import numpy as np

from collections import defaultdict

import random


class DataLoader(object):
    MIN_COUNT_TO_CONSIDER = 3

    def __init__(self, path):
        self.path = path

        self.datas_train = []
        self.datas_test = []

        self.tokens = []
        self.token_to_id = dict()

        self._load_train_vocab()


    def _load_train_vocab(self):
        tokens_count = defaultdict(int)

        with open("{}/train/data.txt".format(self.path), "r") as file:
            for line in file:
                tokens = line.split()
                self.datas_train.append(tokens)

                for token in tokens:
                    tokens_count[token] += 1


        for token in tokens_count:
            if tokens_count[token] >= self.MIN_COUNT_TO_CONSIDER:
                self.tokens.append(token)

        self.tokens += ["_PAD_", "_EOS_", "_UNK_"]
        self.token_to_id = dict(zip(tokens, range(len(tokens))))


    def _get_token_id(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]

        return self.token_to_id["_UNK_"]


    def datas_to_matrix(self, datas, max_len=None):
        max_len = max_len or max(map(len, datas))

        mtx = np.zeors([len(datas), max_len]) + self.token_to_id["_PAD_"]

        for irow in range(len(datas)):
            for itoken, token in enumerate(dats[irow]):
                mtx[irow, itoken]  = self._get_token_id(token)

        return mtx


    def get_train_batch(self, batch_size):
        start_index = random.randint(
            0,
            len(self.datas_train) - batch_size - 1
        )

        return datas_to_matrix(
            self.datas_trian[start_index:(start_index + batch_size)]
        )


    def get_vocab(self):
        return self.tokens
