import sys
import numpy as np
from collections import defaultdict
import random

import torch
from torch.autograd import Variable

import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DataLoader(object):
    MIN_COUNT_TO_CONSIDER = 12

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
            print("Train data loading...")
            for line in tqdm.tqdm(file):
                tokens = line.split()
                self.datas_train.append(tokens)

                for token in tokens:
                    tokens_count[token] += 1


        print("Tokens calculation...")
        for token in tqdm.tqdm(tokens_count):
            if tokens_count[token] >= DataLoader.MIN_COUNT_TO_CONSIDER:
                self.tokens.append(token)

        self.tokens += ["_PAD_", "_EOS_", "_UNK_"]
        self.token_to_id = dict(zip(
            self.tokens,
            range(len(self.tokens))
        ))


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
        start_index = random.randint(
            0,
            len(self.datas_train) - batch_size - 1
        )

        return Variable(torch.LongTensor(self.datas_to_matrix(
            self.datas_train[start_index:(start_index + batch_size)]
        ))).to(device)


    def get_vocab_size(self):
        return len(self.tokens)
