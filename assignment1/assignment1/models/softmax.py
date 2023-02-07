"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def softmax(self, inp: np.ndarray) -> np.ndarray:
    
        inp = inp - inp.max()

        return np.exp(inp) / np.sum(np.exp(inp))

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me

        

        return

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        # one hot encoding
        vector_lookup = np.eye(y_train.max() + 1)
        y_train_one_hot = vector_lookup[y_train]

        batch_size = 64
        X_train = np.insert(X_train, X_train.shape[1], 1, axis=1)
        self.w = np.random.randn(X_train.shape[1], y_train_one_hot.shape[1]) * 1e-1
        # training
        for epoch in range(self.epochs):
            
            x_batched = X_train
            y_batched = y_train_one_hot

            for i in range(batch_size):
                
                output = np.dot(x_batched, self.w)
                grad = self.calc_gradient(x_batched, y_batched)

                self.w = self.w - self.lr * grad



    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        return
