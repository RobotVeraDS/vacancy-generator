import torch.nn.functional as F
import numpy as np


class Trainer(object):
    def __init__(self):
        pass


    def train(self, model, optimizer, data_loader, batch_size, num_epochs,
              batches_per_epoch):
        for ind_epoch in range(num_epochs):

            epoch_losses = []
            for ind_batch in range(batches_per_epoch):
                batch = data_loader.get_random_train_batch(batch_size)

                model.zero_grad()
                probas = model(batch)

                loss = F.nll_loss(
                    probas[:, :-1].contiguous().view(-1, data_loader.get_vocab_size()),
                    batch[:, 1:].contiguous().view(-1)
                )

                epoch_losses.append(loss.item())

                loss.backward()
                optimizer.step()

            print("Epoch", ind_epoch, "Loss", np.mean(epoch_losses))
