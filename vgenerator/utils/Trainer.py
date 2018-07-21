import torch.nn.functional as F
import numpy as np
import datetime
import re

import torch
from torch.autograd import Variable


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self):
        pass


    def train(self, model, optimizer, data_loader, batch_size, num_epochs,
              batches_per_epoch, save_every, print_every, check_every, seeds,
              test_max_len=50, test_temperature=1.0):
        num_tokens = data_loader.get_vocab_size()

        for ind_epoch in range(num_epochs):
            epoch_start = datetime.datetime.now()
            epoch_losses = []
            for ind_batch in range(batches_per_epoch):
                batch = data_loader.get_random_train_batch(batch_size)

                model.zero_grad()
                probas = model(batch)

                loss = F.nll_loss(
                    probas[:, :-1].contiguous().view(-1, num_tokens),
                    batch[:, 1:].contiguous().view(-1)
                )

                epoch_losses.append(loss.item())

                loss.backward()
                optimizer.step()

            if ind_epoch % save_every == 0:
                self.save_checkpoint(ind_epoch, model, optimizer)

            if ind_epoch % print_every == 0:
                epoch_seconds = round((
                    datetime.datetime.now() - epoch_start
                ).total_seconds(), 2)

                print(
                    "Epoch", ind_epoch + 1,
                    "seconds:", epoch_seconds,
                     "loss:", np.mean(epoch_losses)
                )

            if ind_epoch % check_every == 0:
                for seed in seeds:
                    out = self.generate_sample(
                        model, data_loader, seed, test_max_len, test_temperature
                    )
                    print(re.sub("_PAD_", "", out).strip())

                print()


    def save_checkpoint(self, ind_epoch, model, optimizer):
        file_pattern = re.sub(
            "[ \\-\\:]",
            "_",
            str(datetime.datetime.now())[:19]
        )

        torch.save(
            {
                "epoch": ind_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer" : optimizer.state_dict()
            },
            "checkpoints/{}.pth.tar".format(file_pattern)
        )


    def generate_sample(self, model, data_loader, seed, max_length, temperature):
        mtx = data_loader.datas_to_matrix([seed])
        x = Variable(torch.LongTensor(mtx)).to(device)

        for _ in range(max_length - len(seed)):
            probas = model(x)[:,-1]
            p_next = F.softmax(probas / temperature, dim=-1).cpu().data.numpy()[0]

            next_ind = np.random.choice(data_loader.get_vocab_size(), p=p_next)
            next_ind = Variable(torch.LongTensor([[next_ind]])).to(device)

            x = torch.cat([x, next_ind], dim=1)

        return " ".join([data_loader.tokens[ix] for ix in x.cpu().data.numpy()[0]])
