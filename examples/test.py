from vgenerator.models import Generator
from vgenerator.utils import DataLoader
from vgenerator.utils import Trainer

import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# params

embedding_size = 24
hidden_size = 24
num_layers = 1

lr = 0.01

batch_size = 10
batches_per_epoch = 10
num_epochs = 10

save_every = 5
print_every = 1
check_every = 1

seeds = [x.split() for x in [
    "Менеджер по туризму",
    "Электромеханик по технической поддержке лкс",
    "Юрисконсульт обязанности : работа с договорами",
    "Руководитель отдела продаж",
    "Менеджер по закупкам    должностные обязанности : анализ продаж",
    "Менеджер отдела по работе с партнерами",
    "Директор филиала",
    "Руководитель проекта отдел строительства",
    "Ученый исследователь",
    "Автослесарь",
    "Медицинский представитель",
    "Врач узи"
]]

# job

data_loader = DataLoader("data/sample")

print("Number of unique tokens:", data_loader.get_vocab_size())

model = Generator(
    data_loader.get_vocab_size(),
    embedding_size,
    hidden_size,
    num_layers
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=10
)

trainer = Trainer()

print("Start training")

trainer.train(model, scheduler, data_loader,
              batch_size, num_epochs, batches_per_epoch,
              save_every, print_every, check_every,
              seeds)
