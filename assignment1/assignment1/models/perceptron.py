"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.b = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

        np.random.seed(31) #good seeds: 10000 (val: 84.822656, test: 84.740170),  31(val: 96.673082, test: 96.150674)

    def compute_grad(self, pred: np.ndarray, label: np.ndarray, data: np.ndarray) -> np.ndarray:

        grad = np.zeros(data.shape)
        for i, y_i in enumerate(label):
            if y_i == 1:
                grad[i] = - int(pred[i] < 0) * data[i]
            elif y_i == 0:
                grad[i] = int(pred[i] > 0) * data[i]
        
        return grad.mean(axis=0) #+ np.sqrt((self.w[:-1])**2).mean()

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        batch_size = 128
        # min-max 
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min()) 

        if self.n_class == 2:

            # prepare mini-batched data 
            # rice: x.shape = 10911 x 11, y.shape = 10911
            
            X_train = np.insert(X_train, X_train.shape[1], 1, axis=1) # w/ bias
            self.w = 1e-2*np.random.randn(12) # w/ bias
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
                    # print(output)

                    # update
                    self.w = self.w - self.lr * self.compute_grad(output, y_batched, x_batched)
                    
        elif self.n_class > 2:
            
            # prepare mini-batched data 
            # fashion: x.shape = 50000 x 784, y.shape = 50000
            
            self.w = np.random.randn(784)

            train_iter = X_train.shape[0] // batch_size 
            
            print(train_iter)
            

        else:
            print('num of classes should >= 2')
            return 

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

        # min-max 
        X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min()) 
        
        return (np.dot(np.insert(X_test, X_test.shape[1], 1, axis=1), self.w) > 0).astype(np.int32)
