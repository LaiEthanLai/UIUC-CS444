# Extra Credit: Implement the same network in PyTorch with autograd and compare the behavior (training curves, computational requirements, output quality) of your numpy and PyTorch networks. This is the code for evaluating the neural network.

import torch
import torch.nn as nn
import os, imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from train import mlp
size = 128

    

    
# Create the mappings dictionary of matrix B -  you will implement this
def get_B_dict(var_list):
  B_dict = {}
  B_dict['none'] = None
  
  # add B matrix for basic, gauss_1.0, gauss_10.0, gauss_100.0
  # TODO implement this
  B_dict['basic'] = np.eye(2)
  for i in var_list:
    B_dict['gauss_{}'.format(i)] = np.random.normal(0, i, (256, 2))

  return B_dict

def input_mapping(x, B):
    if B is None:
        # "none" mapping - just returns the original input coordinates
        return x
    else:
        # "basic" mapping and "gauss_X" mappings project input features using B
        # TODO implement this
        proj = 2 * np.pi * np.matmul(x, B.T)

    return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)

# Apply the input feature mapping to the train and test data - already done for you
def get_input_features(B_dict, mapping, train_data, test_data, output_size):
    # mapping is the key to the B_dict, which has the value of B
    # B is then used with the function `input_mapping` to map x  

    y_train = train_data[1].reshape(-1, output_size)
    y_test = test_data[1].reshape(-1, output_size)
    X_train = input_mapping(train_data[0].reshape(-1, 2), B_dict[mapping])
    X_test = input_mapping(test_data[0].reshape(-1, 2), B_dict[mapping])
    return X_train, y_train, X_test, y_test


def get_image(size=512, \
              image_url='https://bmild.github.io/fourfeat/img/lion_orig.png'):

    # Download image, take a square crop from the center  
    img = imageio.v2.imread(image_url)[..., :3] / 255.
    c = [img.shape[0]//2, img.shape[1]//2]
    r = 256
    img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]

    if size != 512:
        img = cv2.resize(img, (size, size))

    # Create input pixel coordinates in the unit square
    coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    x_test = np.stack(np.meshgrid(coords, coords), -1)
    test_data = [x_test, img]
    train_data = [x_test[::2, ::2], img[::2, ::2]]
  
    return train_data, test_data

def mse(y, p):
  # TODO implement this
  # make sure it is consistent with your implementation in neural_net.py
  return np.mean((y-p)**2)

def psnr(y, p):
  # TODO implement this
  return -10 * np.log10(2.*mse(y, p))

def to_tensor(x, device='cuda', dtype=torch.float32):

    return torch.from_numpy(x).to(device=device, dtype=dtype)

def plot_training_curves(train_loss, test_loss, train_psnr, test_psnr):

  # plot the training loss
  plt.subplot(2, 1, 1)
  plt.plot(train_loss, label='train')
  plt.plot(test_loss, label='test')
  plt.title('MSE history')
  plt.xlabel('Iteration')
  plt.ylabel('MSE Loss')
  plt.legend()

  # plot the training and testing psnr
  plt.subplot(2, 1, 2)
  plt.plot(train_psnr, label='train')
  plt.plot(test_psnr, label='test')
  plt.title('PSNR history')
  plt.xlabel('Iteration')
  plt.ylabel('PSNR')
  plt.legend()

  plt.tight_layout()
  plt.show()

def plot_reconstruction(p, y_test):
  p_im = p.reshape(size,size,3)
  y_im = y_test.reshape(size,size,3)

  plt.figure(figsize=(12,6))

  # plot the reconstruction of the image
  plt.subplot(1,2,1), plt.imshow(p_im), plt.title("reconstruction")

  # plot the ground truth image
  plt.subplot(1,2,2), plt.imshow(y_im), plt.title("ground truth")

  print("Final Test MSE", mse(y_test, p))
  print("Final Test psnr",psnr(y_test, p))

  plt.show()

def plot_reconstruction_progress(predicted_images, y_test, N=8):
  total = len(predicted_images)
  step = total // N
  plt.figure(figsize=(24, 4))

  # plot the progress of reconstructions
  for i, j in enumerate(range(0,total, step)):
      plt.subplot(1, N, i+1)
      plt.imshow(predicted_images[j].reshape(size,size,3))
      plt.axis("off")
      plt.title(f"iter {j}")

  # plot ground truth image
  plt.subplot(1, N+1, N+1)
  plt.imshow(y_test.reshape(size,size,3))
  plt.title('GT')
  plt.axis("off")
  plt.show()

def plot_feature_mapping_comparison(outputs, gt):
  # plot reconstruction images for each mapping
  plt.figure(figsize=(24, 4))
  N = len(outputs)
  for i, k in enumerate(outputs):
      plt.subplot(1, N+1, i+1)
      plt.imshow(outputs[k]['pred_imgs'][-1].reshape(size, size, -1))
      plt.title(k)
  plt.subplot(1, N+1, N+1)
  plt.imshow(gt)
  plt.title('GT')
  plt.show()

  # plot train/test error curves for each mapping
  iters = len(outputs[k]['train_psnrs'])
  plt.figure(figsize=(16, 6))
  plt.subplot(121)
  for i, k in enumerate(outputs):
      plt.plot(range(iters), outputs[k]['train_psnrs'], label=k)
  plt.title('Train error')
  plt.ylabel('PSNR')
  plt.xlabel('Training iter')
  plt.legend()
  plt.subplot(122)
  for i, k in enumerate(outputs):
      plt.plot(range(iters), outputs[k]['test_psnrs'], label=k)
  plt.title('Test error')
  plt.ylabel('PSNR')
  plt.xlabel('Training iter')
  plt.legend()
  plt.show()

def main(args):
    # TODO pick an image and replace the url string
    size = 128
    train_data, test_data = get_image(size) #image_url="https://i.imgur.com/KpRYoOM.png"
    var_list =  [1, 10]
    B_dict = get_B_dict(var_list)

    # outputs = {}
    for mapping in (B_dict):
        out_size  = 3
        size = 128
        X_train, y_train, X_test, y_test = get_input_features(B_dict, mapping, train_data, test_data, out_size)
        in_size = X_train.shape[1]
        
        net = mlp(sizes=[in_size, 256, 256, 256, 256, 256, 256, 256, out_size]).cuda()

        chpt = torch.load((f'chpt_{mapping}_{args.filename}.pt'))['model_state_dict']
        net.load_state_dict(chpt)
        net = net.cuda()
        net.eval()

        
        with open(f'{mapping}_trainloss_{args.filename}.npy', 'rb') as f:
            train_loss = np.load(f,allow_pickle=True)
        with open(f'{mapping}_testloss_{args.filename}.npy', 'rb') as f:
            test_loss = np.load(f,allow_pickle=True)
        with open(f'{mapping}_train_psnr_{args.filename}.npy', 'rb') as f:
            train_psnr = np.load(f,allow_pickle=True)
        with open(f'{mapping}_test_psnr_{args.filename}.npy', 'rb') as f:
            test_psnr = np.load(f,allow_pickle=True)

        with torch.no_grad():
            plot_reconstruction(net(to_tensor(X_test)).cpu().numpy(), y_test)
            plot_training_curves(train_loss, test_loss, train_psnr, test_psnr)
        
        del net

def parse():
    parser = ArgumentParser()
    parser.add_argument('--filename', '--f', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    main(args)