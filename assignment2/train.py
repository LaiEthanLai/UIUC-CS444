# Extra Credit: Implement the same network in PyTorch with autograd and compare the behavior (training curves, computational requirements, output quality) of your numpy and PyTorch networks. This is the code for training the neural network.

import torch
import torch.nn as nn
import os, imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

size = 128

def parse():
    parser = ArgumentParser()
    parser.add_argument('--filename', '--f', type=str)
    parser.add_argument('--epoch', '--e', default=1000)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--optim', default='AdamW')
    parser.add_argument('--vars', default=[1, 10], nargs='+')
    parser.add_argument('--auto_cast', '--a', action='store_true', help='PyTorch will cast tensor dtype to float16, which increases the training speed')
    return parser.parse_args()

class mlp_block(nn.Module):
    def __init__(self, in_, out_) -> None:
        super().__init__()
        
        self.b = nn.Sequential(
            nn.Linear(in_, out_),
            nn.BatchNorm1d(out_)
        )

    def forward(self, x, residual):
        return self.b(x) + x if residual else self.b(x)

class mlp(nn.Module):
    def __init__(self, sizes) -> None:
        super().__init__()
        
        self.linears = []
        for i in range(len(sizes)-1):
            self.linears.append(mlp_block(sizes[i], sizes[i+1]))
        self.linears = nn.ParameterList(self.linears)
        self.sizes = sizes
    
    def forward(self, x):
        
        for idx, _ in enumerate(self.linears):
            x = self.linears[idx](x, self.sizes[idx]==self.sizes[idx+1])
            if idx != len(self.linears)-1:
                x = nn.LeakyReLU(0.01)(x)
            
        x = torch.sigmoid(x)
        
        return x
    
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
    size = 128
    # TODO pick an image and replace the url string
    train_data, test_data = get_image(size) #image_url="https://i.imgur.com/KpRYoOM.png"

    lr = args.lr
    epochs = args.epoch
    batch_size = args.batch_size

    var_list =  args.vars
    B_dict = get_B_dict(var_list)


    # outputs = {}
    for mapping in (B_dict):

        print('training: {}........................'.format(mapping))
        
        out_size = size = 3
        X_train, y_train, X_test, y_test = get_input_features(B_dict, mapping, train_data, test_data, out_size)
        in_size = X_train.shape[1]
        
        net = mlp(sizes=[in_size, 256, 256, 256, 256, 256, 256, 256, out_size]).cuda()
        optimizer = None
        if args.optim == 'AdamW':
            optimizer = torch.optim.AdamW(params=net.parameters(), lr=lr, weight_decay=1e-3)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
        elif args.optim == 'SGD':
            optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, momentum=0.9)
        
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 500, 800], gamma=0.1, last_epoch=- 1, verbose=False)
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)

        train_loss = np.zeros(epochs)
        train_psnr = np.zeros(epochs)
        test_psnr = np.zeros(epochs)
        test_loss = np.zeros(epochs)
        predicted_images = np.zeros((epochs, y_test.shape[0], y_test.shape[1]))

        best = 1e3
        for epoch in (range(epochs)):
            
            index = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[index]
            y_train_shuffled = y_train[index]

            num_step = X_train.shape[0] // batch_size 

            loss_ = 0.0
            psnr_ = 0.0
            net.train()
            for i in range(num_step):
                if args.auto_cast:
                    with torch.autocast(device_type='cuda'):
                        output = net(to_tensor(X_train_shuffled[i*batch_size : (i+1)*batch_size]))
                        loss = criterion(to_tensor(y_train_shuffled[i*batch_size : (i+1)*batch_size]), output)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    output = net(to_tensor(X_train_shuffled[i*batch_size : (i+1)*batch_size]))
                    loss = criterion(to_tensor(y_train_shuffled[i*batch_size : (i+1)*batch_size]), output)
                    loss_ += loss.item()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    psnr_ += psnr(y_train_shuffled[i*batch_size : (i+1)*batch_size], output.cpu().numpy())
            
            train_loss[epoch] = loss_ / num_step
            train_psnr[epoch] = psnr_ / num_step

            if (epoch+1)%100 == 0:
                print(f'training of epoch {epoch+1}: loss = {train_loss[epoch]}, psnr = {train_psnr[epoch]}')

            with torch.no_grad():
                net.eval()
                output = net(to_tensor(X_test))
                test_psnr[epoch] = psnr(y_test, output.cpu().numpy())
                predicted_images[epoch] = output.cpu().numpy()
                test_loss[epoch] = np.mean((predicted_images[epoch]-y_test)**2)
                if (epoch+1)%100 == 0:
                    print(f'testing of epoch {epoch+1}: loss = {test_loss[epoch]}, psnr = {test_psnr[epoch]}')


                if best > test_loss[epoch]:
                    torch.save({
                        'model_state_dict': net.state_dict()
                    }, f'chpt_{mapping}_{args.filename}.pt')
                    print(f'save chpt at {epoch+1} of mapping: {mapping}, prev loss {best} -> current best {test_loss[epoch]}')
                    best = test_loss[epoch]
            
            lr_sched.step()
        
        with open(f'{mapping}_trainloss_{args.filename}.npy', 'wb') as f:
            np.save(f, train_loss)
        with open(f'{mapping}_testloss_{args.filename}.npy', 'wb') as f:
            np.save(f, test_loss)
        with open(f'{mapping}_train_psnr_{args.filename}.npy', 'wb') as f:
            np.save(f, train_psnr)
        with open(f'{mapping}_test_psnr_{args.filename}.npy', 'wb') as f:
            np.save(f, test_psnr)

        del net
        del optimizer
        del lr_sched

if __name__ == '__main__':
    args = parse()
    main(args)