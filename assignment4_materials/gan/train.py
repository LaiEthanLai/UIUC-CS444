import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img
from gan.losses import compute_gradient_penalty, w_gan_disloss

import torch

from random import random

def getRandom(low: float, high: float) -> float:

    return random() * (high - low) + low

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, gan_type, show_every=250, 
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None, train_every=1, l_gp=10, soft_label=False, save_every=None):
    """
    Train loop for GAN.
    
    The loop will consist of two steps: a discriminator step and a generator step.
    
    (1) In the discriminator step, you should zero gradients in the discriminator 
    and sample noise to generate a fake data batch using the generator. Calculate 
    the discriminator output for real and fake data, and use the output to compute
    discriminator loss. Call backward() on the loss output and take an optimizer
    step for the discriminator.
    
    (2) For the generator step, you should once again zero gradients in the generator
    and sample noise to generate a fake data batch. Get the discriminator output
    for the fake data batch and use this to compute the generator loss. Once again
    call backward() on the loss and take an optimizer step.
    
    You will need to reshape the fake image tensor outputted by the generator to 
    be dimensions (batch_size x input_channels x img_size x img_size).
    
    Use the sample_noise function to sample random noise, and the discriminator_loss
    and generator_loss functions for their respective loss computations.
    
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """
    iter_count = 0
    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))
        for idx, (x, _) in enumerate(train_loader):
            _, input_channels, img_size, _ = x.shape
            
            real_images = preprocess_img(x).to(device)  # normalize
            
            # Store discriminator loss output, generator loss output, and fake image output
            # in these variables for logging and visualization below
            d_error = None
            g_error = None
            fake_images = None
            
            ####################################
            #          YOUR CODE HERE          #
            ####################################
            
            # train discriminator
            D_solver.zero_grad()
            
            # real
            real_label = torch.ones(batch_size).to(device, torch.float32) 
            if soft_label:
               real_label *= getRandom(0.7, 1.2)
            real_logit = D(real_images)
            d_error_real = discriminator_loss(real_logit, real_label) if discriminator_loss != w_gan_disloss else discriminator_loss(-real_logit, None)
            d_error_real.backward()
            
            # fake
            sample = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(sample).reshape(batch_size, input_channels, img_size, img_size)
            
            
            fake_label = torch.zeros(batch_size).to(device, torch.float32) 
            if soft_label:
              fake_label += getRandom(0, 0.3)
            
            fake_logit = D(fake_images.detach())

            d_error_fake = discriminator_loss(fake_logit, fake_label)
            d_error_fake.backward()
            
            d_error_gp = 0
            if discriminator_loss ==  w_gan_disloss:
              d_error_gp = compute_gradient_penalty(D, real_images.data, fake_images.data) * l_gp
              d_error_gp.backward()

            d_error = d_error_real + d_error_fake + d_error_gp

            D_solver.step()

            # train generator
            G_solver.zero_grad()

            if (idx+1)%train_every == 0:
              g_error = generator_loss(D(fake_images))
              g_error.backward()
              G_solver.step()

            ##########       END      ##########
            
            # Logging and output visualization
            if (iter_count % show_every == 0):
                if g_error:
                  print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))
                else:
                  print('Iter: {}, D: {:.4}, G is not trained in this iter'.format(iter_count,d_error.item()))
                with torch.no_grad():
                  fake_images = G(sample).reshape(batch_size, input_channels, img_size, img_size)
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels!=1)
                plt.show()
                print()
            iter_count += 1

            if (iter_count % save_every == 0):
              torch.save(G.state_dict(), f'{gan_type}_{iter_count}.pt')
              torch.save(D.state_dict(), f'{gan_type}_{iter_count}.pt')

def train_wgan():
  pass
#    if discriminator_loss ==  w_gan_disloss:
#               d_error += compute_gradient_penalty(D, real_images.detach(), gen_image.detach()) * l_gp