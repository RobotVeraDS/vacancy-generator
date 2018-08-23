from vgenerator.models import Generator
from vgenerator.utils import DataLoader
from vgenerator.utils import Trainer
from vgenerator.optimizer import Optimizer

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse

# params

PROJECT_PATH = 'usa/'
DATA_PATH = 'data/sample'

embedding_size = 24
hidden_size = 24
num_layers = 1

lr = 0.01

batch_size = 10
batches_per_epoch = 10
num_epochs = 100

save_every = 2
print_every = 1
check_every = 20

network_type = "word"

gpu_number = 0

DataLoader.MAX_VOCAB_SIZE = 30000
DataLoader.MIN_TOKEN_COUNT = 10

test_max_len = 140

# job
def main():

    parser = argparse.ArgumentParser('Modify params of model.')
    parser.add_argument('-pp', '--project_path', required=False, type=str,
                        default=PROJECT_PATH)
    parser.add_argument('-dp', '--data_path', required=False, type=str,
                        default=DATA_PATH)
    parser.add_argument('-es', '--embedding_size', required=False, type=int,
                        default=embedding_size)
    parser.add_argument('-hs', '--hidden_size', required=False, type=int,
                        default=hidden_size)
    parser.add_argument('-nl', '--num_layers', required=False, type=int,
                        default=num_layers)
    parser.add_argument('-lr', '--lr', required=False, type=float,
                        default=lr)
    parser.add_argument('-bs', '--batch_size', required=False, type=int,
                        default=batch_size)
    parser.add_argument('-bpe', '--batches_per_epoch', required=False, type=int,
                        default=batches_per_epoch)
    parser.add_argument('-ne', '--num_epochs', required=False, type=int,
                        default=num_epochs)
    parser.add_argument('-se', '--save_every', required=False, type=int,
                        default=save_every)
    parser.add_argument('-pe', '--print_every', required=False, type=int,
                        default=print_every)
    parser.add_argument('-ce', '--check_every', required=False, type=int,
                        default=check_every)
    parser.add_argument('-nt', '--network_type', required=False, type=str,
                        default=network_type)
    parser.add_argument('-gpu', '--gpu_number', required=False, type=int,
                        default=gpu_number)
    parser.add_argument('-mvs', '--max_vocab_size', required=False, type=int,
                        default=DataLoader.MAX_VOCAB_SIZE)
    parser.add_argument('-mtc', '--max_token_count', required=False, type=int,
                        default=DataLoader.MIN_TOKEN_COUNT)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_number)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    data_loader = DataLoader(args.data_path, device, type=args.network_type)

    seeds = [data_loader._get_tokens(x) for x in [

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


    print("Number of unique tokens:", data_loader.get_vocab_size())

    model = Generator(
        data_loader.get_vocab_size(),
        args.embedding_size,
        args.hidden_size,
        args.num_layers
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optim,
        mode="min",
        factor=0.1,
        patience=2
    )

    optimizer = Optimizer(optim, scheduler)

    trainer = Trainer(device, args.project_path)

    print("Start training")

    trainer.train(model, optimizer, data_loader,
                  args.batch_size, args.num_epochs, args.batches_per_epoch,
                  args.save_every, args.print_every, args.check_every,
                  seeds)
if __name__ == '__main__':
    main()
