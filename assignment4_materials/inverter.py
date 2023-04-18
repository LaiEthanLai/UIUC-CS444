import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
import torchvision.transforms as transforms

from gan.models import Generator, Discriminator
from gan.losses import w_gan_disloss, w_gan_genloss, compute_gradient_penalty
from gan.utils import sample_noise
from GAN_domain_encoder import In_domain_encoder

from argparse import ArgumentParser
from collections import OrderedDict

from PIL import Image
import yaml

def main(args):
    
    config = yaml.load(open(args.config, 'r'), yaml.FullLoader)
    device = config['device']

    # load img to be inverted
    target = Image.open(config['target_img'])
    target = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(int(1.15 * config['target_shape'])),
        transforms.RandomCrop(config['target_shape']),
    ])(target)

    # Pytorch related
    if 'G_weight' in config:
        g_weight = torch.load(config['G_weight'])
    else:
        print('Plz provide a trained wight!!!')
        raise NotImplementedError
    G = Generator(noise_dim=config['noise_dim']).to(device)
    G.load_state_dict(g_weight)
    for param in G.parameters():
        param.requires_grad = False

    if 'enc_weight' in config:
        enc_weight = torch.load(config['enc_weight'])
        encoder = In_domain_encoder(noise_dim=config['noise_dim']).to(device)
        encoder.load_state_dict(enc_weight)
        for param in encoder.parameters():
            param.requires_grad = False
    
    if 'l_percep' in config:
        vgg_model = vgg16(pretrained=True)
        vgg_model.eval()
        for param in vgg_model.parameters():
            param.requires_grad = False

    

def parsing():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='naive.yaml')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parsing()
    main(args)
