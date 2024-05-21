# Assignment 2: Multi-Layer Neural Networks
## Implement the neural network components with forward pass and backward pass
We implement the behaviors of components needed to build up and train a neural network, e.g., linear layers, activation functions, and optimizers. The implementations can be found in `models/neural_net.py`.

Note: Use `develop_neural_network.ipynb` to check the correctness of implementation of each components.

## Image Reconstruction
In this paper, the author researched the problem of reconstructing an image via an MLP model. Specifically, the MLP takes the coordinates of a pixel as input, and outputs the corresponding RGB values of the pixel. The authors devised a fourier feature mapping to enhance the reconstruction of high frequency components of the image. Please refer to the details in this [paper](https://bmild.github.io/fourfeat/)).

In `neural_network.ipynb`, we implement the fourier feature mapping and the training code.

## Result
Organizing...
