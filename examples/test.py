from vgenerator.models import Generator
from vgenerator.utils import DataLoader
from vgenerator.utils import Trainer

import torch

data_loader = DataLoader("data/sample")

embedding_size = 100
hidden_size = 100
num_layers = 2

model = Generator(
    data_loader.get_vocab_size(),
    embedding_size,
    hidden_size,
    num_layers
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

batch_size = 10
batches_per_epoch = 5
num_epochs = 100

save_every = 10
print_every = 2

trainer = Trainer()

trainer.train(model, optimizer, data_loader,
              batch_size, num_epochs, batches_per_epoch,
              save_every, print_every)
