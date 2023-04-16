import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

from gan.models import Generator
from argparse import ArgumentParser
from collections import OrderedDict

from PIL import Image

class InvertedGenereator(nn.Module):
    def __init__(self, noise_dim: int, G: nn.Module, trained_weight: OrderedDict = None, output_channels = 3):
        super().__init__(noise_dim, output_channels)

        self.latent_code = torch.randn(1, noise_dim, 1, 1)
        self.latent_code = nn.parameter.Parameter(data=self.latent_code, requires_grad=True)

        self.gen = G
       
        self.gen.load_state_dict(trained_weight)
        for i in self.gen.parameters():
            i.requires_grad = False
        

    def forward(self, x=None):

        return self.gen(self.latent_code)

def main(args):
    
    if args.weight:
        weight = torch.load(args.weight)
    else:
        print('Plz provide a trained wight!!!')
        raise NotImplementedError
    
    
    device = args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    print(f'train on {device}')
    
    model = InvertedGenereator(noise_dim=args.noise_dim).to(device)
    target = Image(args.target)
    target = ToTensor()(target).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    for i in args.iters:

        output = model()
        loss = 0.5 * ((output - target) ** 2)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'{args.save_path}.pt')

def parsing():
    parser = ArgumentParser()
    parser.add_argument('--noise_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_path', type=str, default='inverted_GAN')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parsing()
    main(args)
