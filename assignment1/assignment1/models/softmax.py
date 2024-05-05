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
        self.w = None  
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

        np.random.seed(42)

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
        
        return np.matmul(y_train.T, X_train) / X_train.shape[0]

    def min_max(self, x: np.ndarray) -> np.ndarray:

        return (x - x.min()) / (x.max() - x.min())

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        # one hot encoding
        vector_lookup = np.eye(y_train.max() + 1)
        y_train_one_hot = vector_lookup[y_train]

        # label smoothing
        # epsilon = 5e-3
        # y_train_one_hot = np.abs(y_train_one_hot - epsilon) / (X_train.shape[0])
        # print(y_train_one_hot[:, y_train])
        # # y_train_one_hot[y_train_one_hot.argmax(axis=1)] *= (X_train.shape[0])

        # print(y_train_one_hot.max())

        batch_size = 64
        X_train = self.min_max(X_train)
        X_train = np.insert(X_train, X_train.shape[1], 1, axis=1)
        self.w = np.random.randn(y_train_one_hot.shape[1], X_train.shape[1]) * 5e-2
        self.w[:, -1] = 0
        train_iter = X_train.shape[0] // batch_size # neglect last X_train.shape[0] % batch_size data
        # training
        for epoch in range(self.epochs):
            
            # shuffle data (https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison)
            shuffled_index = np.random.permutation(X_train.shape[0])
            X_train = X_train[shuffled_index]
            y_train_one_hot = y_train_one_hot[shuffled_index]

            for i in range(train_iter):
                
                x_batched = X_train[i*batch_size:(i+1)*batch_size]
                y_batched = y_train_one_hot[i*batch_size:(i+1)*batch_size]
                
                output = np.dot(x_batched, self.w.T)
                output = self.softmax(output)
                
                grad = self.calc_gradient(x_batched, (output - y_batched))
                self.w = self.w - self.lr * (grad + self.reg_const * (self.w) / batch_size)
            
            if (epoch+1)%20 == 0:
                self.lr *= 0.92


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

        X_test = self.min_max(X_test)
        X_test = np.insert(X_test, X_test.shape[1], 1, axis=1)
        
        return  self.softmax(np.dot(X_test, self.w.T)).argmax(axis=1)
