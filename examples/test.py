from vgenerator.models import Generator
from vgenerator.utils import DataLoader
from vgenerator.utils import Trainer

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# params

embedding_size = 100
hidden_size = 100
num_layers = 2

lr = 0.01

batch_size = 10
batches_per_epoch = 5
num_epochs = 100

save_every = 10
print_every = 2

seeds = [x.split() for x in [
    "менеджер по туризму",
    "электромеханик по технической поддержке лкс",
    "юрисконсульт обязанности : работа с договорами",
    "руководитель отдела продаж",
    "менеджер по закупкам    должностные обязанности : анализ продаж",
    "менеджер отдела по работе с партнерами",
    "директор филиала",
    "руководитель проекта отдел строительства",
    "ученый исследователь",
    "автослесарь",
    "медицинский представитель",
    "врач узи"
]]

# job

data_loader = DataLoader("data/sample")

model = Generator(
    data_loader.get_vocab_size(),
    embedding_size,
    hidden_size,
    num_layers
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

trainer = Trainer()

trainer.train(model, optimizer, data_loader,
              batch_size, num_epochs, batches_per_epoch,
              save_every, print_every, seeds)
