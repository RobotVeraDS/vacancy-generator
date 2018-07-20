from vgenerator.models import Generator
from vgenerator.utils import DataLoader

model = Generator(10, 10, 10, 2)

data_loader = DataLoader("data/sample")

tokens = data_loader.get_vocab()
print(tokens[:10])
