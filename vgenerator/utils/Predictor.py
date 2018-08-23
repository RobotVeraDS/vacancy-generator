import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import re

class Predictor(object):

    def __init__(self, device, type_):
        self.device = device
        self.type_ = type_

    def __call__(self, model, data_processor, seed, max_length=100, temperature=1.0):
        mtx = data_processor.datas_to_matrix([seed])
        x, hidden = Variable(torch.LongTensor(mtx)).to(self.device), None

        # path through model all data except last word
        if len(x[0]) > 1:
            _, hidden = model(x[:, :-1], hidden)

        for _ in range(max_length - len(seed)):
            # add last word and calc next state
            probas, hidden = model(x[:, -1:], hidden)

            last_probas = probas[:, -1]
            p_next = F.softmax(last_probas / temperature, dim=-1).cpu().data.numpy()[0]

            next_ind = np.random.choice(data_processor.get_vocab_size(), p=p_next)
            next_ind = Variable(torch.LongTensor([[next_ind]])).to(self.device)

            x = torch.cat([x, next_ind], dim=1)

        sep = " " if self.type_ == "word" else ""
        out = sep.join([data_processor.tokens[ix] for ix in x.cpu().data.numpy()[0]])
        #out = re.sub("_PAD_", "", out).strip()
        return out
