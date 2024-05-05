"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
        batch_size: int
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        self.batch_size = batch_size

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        self.m = {}
        self.v = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.m[f'W{i}'] = 0
            self.v[f'W{i}'] = 0

            self.params["b" + str(i)] = np.zeros(sizes[i])
            self.m[f'b{i}'] = 0
            self.v[f'b{i}'] = 0

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me

        return

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.where(X > 0, X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return np.where(X > 0, 1, 0)

    def leaky_relu(self, X: np.ndarray, slope: float = 0.01) -> np.ndarray: 

        return np.where(X > 0, X, slope * X)

    def leaky_relu_grad(self, X: np.ndarray, slope: float = 0.01) -> np.ndarray: 

        return np.where(X > 0, 1, slope)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
      # TODO ensure that this is numerically stable
      return np.piecewise(x, [x > 0], [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))])

    def sigmoid_grad(self, x: np.ndarray) -> np.ndarray:
      # TODO ensure that this is numerically stable
      sig = self.sigmoid(x)
      return sig * (1 - sig)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
      # TODO implement this
      return np.mean(((y-p)**2))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        self.outputs['act_0'] = X

        for i in range(1, self.num_layers + 1):
            
            X = np.dot(X, self.params[f'W{i}']) + self.params[f'b{i}']
            self.outputs[f'linear_{i}'] = X
            X = self.leaky_relu(X) if i != self.num_layers else self.sigmoid(X)
            self.outputs[f'act_{i}'] = X

        return X

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        
        loss = self.mse(y, self.outputs[f'act_{self.num_layers}'])
        back = -2*(y - self.outputs[f'act_{self.num_layers}']) / 3
        for i in range(self.num_layers, 0, -1):

            c_z = back * self.leaky_relu_grad(self.outputs[f'linear_{i}']) if i != self.num_layers else back * self.sigmoid_grad(self.outputs[f'linear_{i}'])
            self.gradients[f'b{i}'] = np.mean(c_z, axis=0)
            self.gradients[f'W{i}'] = np.dot(self.outputs[f'act_{i-1}'].T, c_z) / self.batch_size
            back = np.dot(c_z, self.params[f'W{i}'].T)

        return loss

    def update(
        self,
        t,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay = 0
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        
        if self.opt == 'Adam':
            # dw += self.weight_decay * weight
            for key in self.params.keys():

                self.m[key] = b1 * self.m[key] + (1 - b1) * self.gradients[key] # momentum
                self.v[key] = b2 * self.v[key] + (1 - b2) * (self.gradients[key]**2)

                m_corrected = self.m[key] / (1 - b1**t)
                v_corrected = self.v[key] / (1 - b2**t)

                self.params[key] -= lr * (m_corrected / ((v_corrected**0.5) + eps) + self.params[key] * weight_decay)

        elif self.opt == 'SGD':
            # dw += self.weight_decay * weight
            for key in self.params.keys():

                self.m[key] = b1 * self.m[key] + (1 - b1) * self.gradients[key] # momentum
                self.params[key] -= lr * (self.m[key] + self.params[key] * weight_decay)

        else:
            raise NotImplementedError