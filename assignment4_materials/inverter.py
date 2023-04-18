import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
import torchvision.transforms as transforms
from torchvision.utils import save_image

from gan.models import Generator, Discriminator
from gan.losses import w_gan_disloss, w_gan_genloss, compute_gradient_penalty
from gan.utils import sample_noise
from GAN_domain_encoder import In_domain_encoder, freeze_model

from argparse import ArgumentParser
from collections import OrderedDict

from PIL import Image
import yaml

def loss_fn(target: torch.Tensor, prediction: torch.Tensor, kind = 'default') -> float:
    '''
    kind: 'default' or int (L-kind norm)
    e.g. kind == 2 means L-2 norm
    '''
    return torch.mean((target - prediction) ** 2) if kind == 'default' else torch.norm((target - prediction), p=kind)


def cal_loss(config: dict, code: torch.Tensor, target: torch.Tensor, G: nn.Module, feature_extractor: nn.Module=None, encoder: nn.Module=None) -> float:

    reconstructed = G(code)
    if feature_extractor is None and encoder is None:
        return loss_fn(target, reconstructed)
    else:
        return loss_fn(target, reconstructed) + config['l_percep'] * loss_fn(feature_extractor(target), feature_extractor(reconstructed)) + config['l_dom'] * loss_fn(encoder(reconstructed), code)
        

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
    freeze_model(G)

    if 'enc_weight' in config:
        enc_weight = torch.load(config['enc_weight'])
        encoder = In_domain_encoder(noise_dim=config['noise_dim']).to(device)
        encoder.load_state_dict(enc_weight)
        freeze_model(enc_weight)
    
    if 'l_percep' in config:
        vgg_model = vgg16(pretrained=True)
        freeze_model(vgg_model)

    z = encoder(target)
    optimizer = optim.Adam([z], lr=config['lr']) 

    for i in config['iter']:

        loss = cal_loss(config, z, target, G, vgg_model, encoder) if 'enc_weight' in config else cal_loss(config, z, target)
        loss.backward()
        optimizer.step()

    save_image(G(z), f"{config['save_path']}.jpg")
    

def parsing():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='naive.yaml')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parsing()
    main(args)
