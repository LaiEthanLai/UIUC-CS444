"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def compute_grad(self, pred: np.ndarray, label: np.ndarray, data: np.ndarray) -> np.ndarray:

        # grad = np.zeros(data.shape)
        # for i, y_i in enumerate(label):
        #     if y_i == 1:
        #         grad[i] = -self.sigmoid(-pred[i]) * data[i]
        #     elif y_i == 0:
        #         grad[i] = self.sigmoid(pred[i]) * data[i]

        label = label * 2 - 1 # {0, 1} -> {1, -1}
        pred = label * pred
        grad = self.sigmoid(pred * -1)[:, None] * data * label[:, None]  # [:, None] for Python broadcasting

        return grad.mean(axis=0) * -1

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        out = np.zeros_like(z)
        for idx, item in enumerate(z):
            out[idx] = 1 / (1 + np.exp(-1*item)) if item >= 0 else np.exp(item) / (np.exp(item) + 1)

        return out
        

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        
        batch_size = 32
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min()) 

        # prepare mini-batched data 
        # rice: x.shape = 10911 x 11, y.shape = 10911
        
        X_train = np.insert(X_train, X_train.shape[1], 1, axis=1) # w/ bias
        self.w = np.random.randn(12) # w/ bias
        self.w[-1] = 0
        
        # train_iter = X_train.shape[0] // batch_size + int(X_train.shape[0] % batch_size != 0) (deal with different weight shapes)
        train_iter = X_train.shape[0] // batch_size # neglect last X_train.shape[0] % batch_size data

        for epoch in range(self.epochs):

            # shuffle data (https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison)
            shuffled_index = np.random.permutation(X_train.shape[0])
            X_train = X_train[shuffled_index]
            y_train = y_train[shuffled_index]

            for i in range(train_iter):
                
                x_batched = X_train[i*batch_size:(i+1)*batch_size]
                y_batched = y_train[i*batch_size:(i+1)*batch_size]

                output = np.dot(x_batched, self.w)
                output = self.sigmoid(output)

                # update
                self.w = self.w - self.lr * self.compute_grad(output, y_batched, x_batched)


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
        return (self.sigmoid(np.dot(np.insert(X_test, X_test.shape[1], 1, axis=1), self.w) > 0)).astype(np.int32)
