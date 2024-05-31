# Assignment 2: Multi-Layer Neural Networks
## Implement the neural network components with forward pass and backward pass
We implement the behaviors of components needed to build up and train a neural network, e.g., linear layers, activation functions, and optimizers. The implementations can be found in `models/neural_net.py`.

Note: Use `develop_neural_network.ipynb` to check the correctness of the implementation of each component.

## Image Reconstruction
In this [paper](https://bmild.github.io/fourfeat/), the author researched the problem of reconstructing an image via an MLP model. Specifically, the MLP takes the coordinates to a pixel as input and outputs the pixel's RGB values. The authors devised a Fourier feature mapping to enhance the reconstruction of high frequency components of the image, increasing the quality of the reconstructed image. 

In `neural_network.ipynb`, we implement the Fourier feature mapping and the training code. `train.py` and `test.py` is the PyTorch version of our implementation for extra credit.

## Result

![](https://i.imgur.com/bIP10FX.png)

![](https://i.imgur.com/X7wHOFl.png)