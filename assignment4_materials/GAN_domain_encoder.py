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

from argparse import ArgumentParser


from PIL import Image

class In_domain_encoder(nn.Module):
    def __init__(self, noise_dim: int) -> None:
        super().__init__()
    
    def forward(self, x):

        pass

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
        weight = torch.load(args.weight)
    else:
        print('Plz provide a trained wight!!!')
        raise NotImplementedError
    
    
    device = args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    print(f'train on {device}')

    cat_train = ImageFolder(root=args.data_root, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(int(1.15 * args.data_size)),
        transforms.RandomCrop(args.data_size),
    ]))

    loader = DataLoader(cat_train, batch_size=args.batch_size, drop_last=True)
    
    domain_encoder = In_domain_encoder(noise_dim=args.noise_dim).to(device)
    D = Discriminator(input_channels=3).to(device)

    G = Generator(noise_dim=args.noise_dim).to(device)
    G.load_state_dict(weight)
    freeze_model(G)

    vgg_model = vgg16(pretrained=True)
    freeze_model(vgg_model)

    domain_optimizer = optim.Adam(domain_encoder.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
    

    for epoch in args.epoch:

        for img in loader:
            
            img = img.to(args.device)
           
            D_optimizer.zero_grad()

            # real
            d_error_real = w_gan_disloss(-D(img), None)
            d_error_real.backward()

            # fake
            fake_img = G(domain_encoder(img))
            d_error_fake = D(fake_img.detach())
            d_error_fake.backward()

            # gradient penalty
            d_error_gp = compute_gradient_penalty(D, img.data, fake_img.data) * args.l_gp
            d_error_gp.backward()

            d_error = d_error_real + d_error_fake + d_error_gp
            D_optimizer.step()

            # update encoder
            domain_optimizer.zero_grad()

            g_error = args.l_genadv * w_gan_genloss(fake_img) + args.l_percep * perceptual(vgg_model, img, fake_img) + nn.functional.mse_loss(img, fake_img) # torch.norm((img - fake_img), p=2) 
            g_error.backward()
            
            domain_optimizer.step()

    torch.save(domain_encoder.state_dict(), f'{args.save_path}.pt')

def parsing():
    parser = ArgumentParser()
    parser.add_argument('--noise_dim', type=int, default=100)
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
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parsing()
    main(args)
