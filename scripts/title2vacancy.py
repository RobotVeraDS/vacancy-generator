from vgenerator.utils import Predictor
from vgenerator.models import Generator
from vgenerator.utils import DataProcessor

import argparse
import torch

#torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_size = 256
hidden_size = 256
num_layers = 1

type_ = 'word'

SUMMARY = '<h3> Summary </h3>'
RESPONSIBILITIES = '<h3> Responsibilities </h3>'
COE = '<h3> Conditions of Employment </h3>'
QUALIFICATIONS = '<h3> Qualifications </h3>'


BLOCKS = {
    SUMMARY: 'checkpoints/usa1/2018_08_23_07_46_07.pth.tar',
    RESPONSIBILITIES: 'checkpoints/usa2/2018_08_24_00_40_14.pth.tar',
    COE: 'checkpoints/usa3/2018_08_23_04_23_32.pth.tar',
    QUALIFICATIONS: 'checkpoints/usa4/2018_08_24_00_44_55.pth.tar',
    }


def coe_style(string):
    if string is not COE:
        return string
    else:
        return '<div class=coe>{}</div>'.format(string)


def style_wraper(string):
    return '<div class=block>{}</div>'.format(string)


def block_wraper(string):
    return '<blockquote>{}</blockquote>'.format(string)


def main():
    parser = argparse.ArgumentParser('Creates a vacancy by title')
    parser.add_argument('-s', '--seed', required=False, nargs='+',
                        default=['Program', 'Analyst'])
    parser.add_argument('-tp', '--temperature', required=False, type=float,
                        default=1.0)
    parser.add_argument('-l', '--maxlength', required=False, type=int,
                        default=128)
    args = parser.parse_args()

    html_text = ['<h2> {} </h2>'.format(" ".join(args.seed))]

    for block in BLOCKS:

        checkpoint = torch.load(BLOCKS[block],
                            map_location= device)

        tokens = checkpoint['tokens']
        data_processor = DataProcessor(tokens)

        model = Generator(data_processor.vocab_size,
                      embedding_size,
                      hidden_size,
                      num_layers
                      ).to(device)

        model.load_state_dict(checkpoint['state_dict'])

        predictor = Predictor(device, type_)
        result = predictor(model, data_processor, args.seed,
                           args.maxlength, args.temperature)

        try:
            html_text.append(style_wraper(coe_style(block) + block_wraper(result.split(block)[1])))
        except IndexError:
            html_text.append(style_wraper(coe_style(block) + block_wraper(result)))

    html_text = '\n\n'.join(html_text)
    with open('vacancy/vacancy.html', 'w') as vacancy:
        vacancy.write(html_text)

    print(html_text)


if __name__ == '__main__':
    main()
