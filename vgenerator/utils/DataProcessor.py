import numpy as np

class DataProcessor(object):
    def __init__(self, tokens, type_='word'):
        self.tokens = tokens
        self.type_ = type_
        self.token_to_id = dict(zip(tokens, range(len(tokens))))
        self.vocab_size = len(tokens)

    def get_vocab_size(self):
        return len(self.tokens)

    def get_token(self, line):
        if self.type_ == 'word':
            return line.split()
        else:
            return list(line)

    def get_token_id(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            return self.token_to_id["_UNK_"]

    def datas_to_matrix(self, datas, max_len=None):
        max_len = max_len or max(map(len, datas))
        mtx = np.zeros([len(datas), max_len]) + self.token_to_id["_PAD_"]
        for irow in range(len(datas)):
            for itoken, token in enumerate(datas[irow]):
                mtx[irow, itoken] = self.get_token_id(token)
        return mtx
