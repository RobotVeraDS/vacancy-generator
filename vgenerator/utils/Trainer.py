import torch.nn.functional as F
import numpy as np
import torch
import datetime
import re


class Trainer(object):
    def __init__(self):
        pass


    def train(self, model, optimizer, data_loader, batch_size, num_epochs,
              batches_per_epoch, save_every, print_every):
        num_tokens = data_loader.get_vocab_size()

        for ind_epoch in range(num_epochs):

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
                print("Epoch", ind_epoch, "loss:", np.mean(epoch_losses))


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
