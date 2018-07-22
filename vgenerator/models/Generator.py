import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_tokens, emb_size, hidden_size, num_layers):
        super(self.__class__, self).__init__()

        self.embedding = nn.Embedding(num_tokens, emb_size)

        self.rnn = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.out = nn.Linear(hidden_size, num_tokens)


    def forward(self, input):
        output, _ = self.rnn(self.embedding(input))
        output = self.out(output)

        return F.log_softmax(output, dim=-1)
