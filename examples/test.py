from vgenerator.models import Generator
from vgenerator.utils import DataLoader
from vgenerator.utils import Trainer

import torch

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

print("Number of unique tokens:", data_loader.get_vocab_size())

model = Generator(
    data_loader.get_vocab_size(),
    embedding_size,
    hidden_size,
    num_layers
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

trainer = Trainer()

print("Start training")

trainer.train(model, optimizer, data_loader,
              batch_size, num_epochs, batches_per_epoch,
              save_every, print_every, check_every,
              seeds)
