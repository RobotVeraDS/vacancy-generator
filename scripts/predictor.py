from vgenerator.utils import Predictor
from vgenerator.models import Generator
from vgenerator.utils import DataProcessor

import argparse
import torch

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_size = 256
hidden_size = 256
num_layers = 1

type_ = 'word'


def main():
    parser = argparse.ArgumentParser('Modify params of predictor')
    parser.add_argument('-p', '--path', required=False, type=str,
                        default='checkpoints/test/2018_08_23_08_07_03.pth.tar')
    parser.add_argument('-s', '--seed', required=False, nargs='+',
                        default=['Program', 'Analyst'])
    parser.add_argument('-tp', '--temperature', required=False, type=float,
                        default=1.0)
    parser.add_argument('-l', '--maxlength', required=False, type=int,
                        default=128)
    args = parser.parse_args()

    checkpoint = torch.load(args.path)
    tokens = checkpoint['tokens']
    data_processor = DataProcessor(tokens)

    model = Generator(data_processor.vocab_size,
                      embedding_size,
                      hidden_size,
                      num_layers
                      ).to(device)

    seeds = [data_processor.get_token(line) for line in [
        'Program Analyst',
        'AIRCRAFT MECHANIC',
        'Cook',
    ]]

    model.load_state_dict(checkpoint['state_dict'])

    predictor = Predictor(device, type_)
    result = predictor(model, data_processor, args.seed,
                           args.maxlength, args.temperature)
    print(result)

if __name__ == '__main__':
    main()
