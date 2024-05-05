# Extra Credit: Starting with your trained model, play around with GAN inversion or latent space traversal techniques to perform image manipulation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, VGG16_Weights
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
from tqdm import tqdm

def loss_fn(target: torch.Tensor, prediction: torch.Tensor, kind = 'default') -> float:
    '''
    kind: 'default' or int (L-kind norm)
    e.g. kind == 2 means L-2 norm
    '''
    return torch.mean((target - prediction) ** 2) if kind == 'default' else torch.norm((target - prediction), p=kind)


def cal_loss(config: dict, code: torch.Tensor, target: torch.Tensor, G: nn.Module, feature_extractor: nn.Module=None, encoder: nn.Module=None) -> float:

    reconstructed = G(code) if G == Generator else G.netG(code)
    loss = 0.0
    if feature_extractor is not None:
        loss += config['l_percep'] * loss_fn(feature_extractor(target), feature_extractor(reconstructed))
    if encoder is not None:      
        loss += config['l_dom'] * loss_fn(encoder(reconstructed), code)                           

    return loss_fn(target, reconstructed) + loss

def main(args):
    
    config = yaml.load(open(args.config, 'r'), yaml.FullLoader)
    device = config['device']

    # load img to be inverted
    target = Image.open(config['target_img'])
    target = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['target_shape'], config['target_shape']), antialias=True),
    ])(target).to(device).unsqueeze(0)
    print(f'target shape: {target.shape}')

    # Pytorch related
    if 'G_weight' in config:
        if config['G_weight'] == 'pgan':
            G = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=True)
        else:
            g_weight = torch.load(config['G_weight'], map_location='cpu')
            G = Generator(noise_dim=config['noise_dim']).to(device)
            G.load_state_dict(g_weight)
            G.proj = nn.Identity().to(device) # our encoder is trained on the projected space
    else:
        print('Plz provide a trained wight!!!')
        raise NotImplementedError
    
    if G == Generator:
        freeze_model(G)
    else:
        freeze_model(G.netG)

    encoder = None
    if 'enc_weight' in config:
        enc_weight = torch.load(config['enc_weight'], map_location='cpu')
        encoder = In_domain_encoder(noise_dim=config['prog_dim']).to(device)
        encoder.load_state_dict(enc_weight)
        freeze_model(encoder)
    
    if 'l_percep' in config:
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(device)
        freeze_model(vgg_model)

    z = encoder(target) if encoder else sample_noise(1, dim=config['noise_dim']).to(device)
    z.requires_grad = True
    optimizer = optim.Adam([z], lr=config['lr']) 

    print(f'mode: domain regularized') if 'enc_weight' in config and 'l_percep' in config else print('mode: naive')
    for i in tqdm(range(config['iter'])):

        loss = cal_loss(config, z, target, G, vgg_model, encoder) if 'enc_weight' in config else cal_loss(config, z, target, G)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        output_img = G(z) if G == Generator else G.netG(z)
        output_img = (output_img + 1.0) / 2.0
    save_image(output_img, f"{config['save_path']}.jpg")
    save_image(target, f"{config['save_target']}.jpg")
    

def parsing():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/naive.yaml')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parsing()
    main(args)
