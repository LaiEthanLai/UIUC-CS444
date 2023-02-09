"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
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

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me

        grad = np.zeros(X_train.shape)
        for i, y_i in enumerate(y_train):
            if y_i == 1:
                grad[i] = - int(pred[i] < 1) * X_train[i]
            elif y_i == 0:
                grad[i] = int(pred[i] > 1) * X_train[i]
        
        return grad.mean(axis=0) 

        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
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

                    # update
                    self.w = self.w - self.lr * (
                        self.calc_gradient(x_batched, y_batched, output) 
                        + self.reg_const * self.w / x_batched.shape[0]) 
                    
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

        X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min()) 
        
        return  (np.dot(np.insert(X_test, X_test.shape[1], 1, axis=1), self.w) > 0).astype(np.int32)