import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from random import random

def getRandom(low: float, high: float) -> float:

    return random() * (high - low) + low

def discriminator_loss(logits_real, label):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    return bce_loss(logits_real.squeeze(), target=label, reduction='mean')

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    real = torch.ones(logits_fake.shape).to(logits_fake.device) 

    ##########       END      ##########
    
    return bce_loss(logits_fake, target=real, reduction='mean')


def ls_discriminator_loss(scores_real, label):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    return 0.5 * torch.mean((scores_real - label) ** 2)

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    label = torch.ones(scores_fake.shape).to(scores_fake.device)
    
    ##########       END      ##########
    
    return 0.5 * torch.mean((scores_fake - label) ** 2)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.empty(real_samples.shape[0], 1, device=real_samples.device).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def w_gan_genloss(d_g_z):
    """
    Computes the Wasserstein loss for the generator.
    
    Inputs:
    - d_g_z: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    return -torch.mean(d_g_z)

def w_gan_disloss(output, dummy):
    """
    Computes the Wasserstein loss for the discriminator.
    
    Inputs:
    - fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    - real: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
  
    return torch.mean(output)