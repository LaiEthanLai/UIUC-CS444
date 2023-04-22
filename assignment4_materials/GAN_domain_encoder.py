import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms

from gan.models import Generator, Discriminator
from gan.losses import w_gan_disloss, w_gan_genloss, compute_gradient_penalty
from gan.utils import preprocess_img

from argparse import ArgumentParser

from tqdm import tqdm, trange
from PIL import Image
from functools import partial
from einops.layers.torch import Reduce

conv3x3 = partial(nn.Conv2d, stride=2, kernel_size=3, padding=1, bias=False)

class In_domain_encoder(nn.Module):
    def __init__(self, noise_dim: int) -> None:
        super().__init__()

        self.enc = nn.Sequential(
            conv3x3(3, 128),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            conv3x3(128, 512),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            conv3x3(512, 1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            conv3x3(1024, 1024),
            conv3x3(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            conv3x3(1024, 1024),
            conv3x3(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(1024, noise_dim)
        )
    
    def forward(self, x):

        return self.enc(x)

def freeze_model(model: nn.Module) -> nn.Module:

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    

def perceptual(model, real: torch.Tensor, fake: torch.Tensor) -> float:
    '''
    compute the L2 norm between the feature of real images and the feature of fake images
    (the features are from VGG16)
    '''
    return nn.functional.mse_loss(model(real), model(fake)) # torch.norm((model(real) - model(fake)), p=2)

def main(args):
    
    if args.weight:
        weight = torch.load(args.weight, map_location='cpu')
    else:
        print('Plz provide a trained wight!!!')
        raise NotImplementedError
    
    
    device = args.device 
    print(f'train on {device}')

    cat_train = ImageFolder(root=args.data_root, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(int(1.15 * args.data_size), antialias=True),
        transforms.RandomCrop(args.data_size),
    ]))

    loader = DataLoader(cat_train, batch_size=args.batch_size, drop_last=True)
    
    domain_encoder = In_domain_encoder(noise_dim=args.prog_dim).to(device)
    D = Discriminator(input_channels=3).to(device)

    G = Generator(noise_dim=args.noise_dim).to(device)
    G.load_state_dict(weight)
    G.proj = nn.Identity().to(device) # our encoder is trained on the projected space
    freeze_model(G)

    vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(device)
    freeze_model(vgg_model)

    domain_optimizer = optim.Adam(domain_encoder.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
    
    with trange(args.epoch) as pbar:
        g_error = 0.0
        for epoch in pbar:

            for idx,(img, _) in enumerate(loader):
                
                img = preprocess_img(img)
                img = img.to(args.device)
            
                D_optimizer.zero_grad()

                # real
                d_error_real = w_gan_disloss(-D(img), None)
                d_error_real.backward()

                # fake
                fake_img = G(domain_encoder(img))
                d_error_fake = w_gan_disloss(D(fake_img.detach()), None)
                d_error_fake.backward()

                # gradient penalty
                d_error_gp = compute_gradient_penalty(D, img.data, fake_img.data) * args.l_gp
                d_error_gp.backward()

                d_error = d_error_real + d_error_fake  + d_error_gp
                D_optimizer.step()

                # update encoder
                if (idx + (epoch * len(loader))) % args.train_every == 0:
                    domain_optimizer.zero_grad()

                    g_error = args.l_genadv * w_gan_genloss(D(fake_img)) + args.l_percep * perceptual(vgg_model, img, fake_img) + nn.functional.mse_loss(img, fake_img) # torch.norm((img - fake_img), p=2) 
                    g_error.backward()

                    domain_optimizer.step()
                
                if (idx + (epoch * len(loader)))%25 == 0:
                    pbar.set_description(f'd loss: {d_error.item()}, g_loss: {g_error.item()}')

    torch.save(domain_encoder.state_dict(), f'{args.save_path}.pt')

def parsing():
    parser = ArgumentParser()
    parser.add_argument('--noise_dim', type=int, default=100)
    parser.add_argument('--prog_dim', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--l_gp', type=float, default=10)
    parser.add_argument('--l_genadv', type=float, default=0.5)
    parser.add_argument('--l_percep', type=float, default=0.3)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_path', type=str, default='domain_encoder')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_size', type=int, default=64)
    parser.add_argument('--train_every', type=int, default=1)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parsing()
    main(args)
