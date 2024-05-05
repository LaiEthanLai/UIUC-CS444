"""Data preprocessing."""

import os
import pickle
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_pickle(f: str) -> Any:
    """Load a pickle file.

    Parameters:
        f: the pickle filename

    Returns:
        the pickled data
    """
    return pickle.load(f, encoding="latin1")


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def get_FASHION_data(
    num_training: int = 49000,
    num_validation: int = 1000,
    num_test: int = 10000,
    normalize: bool = True,
):
    """Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.

    Parameters:
        num_training: number of training images
        num_validation: number of validation images
        num_test: number of test images
        subtract_mean: whether or not to normalize the data

    Returns:
        the train/val/test data and labels
    """
    # Load the raw FASHION data
    X_train, y_train = load_mnist('fashion-mnist', kind='train')
    X_test, y_test = load_mnist('fashion-mnist', kind='t10k')
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask].astype(float)
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask].astype(float)
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask].astype(float)
    y_test = y_test[mask]
    # Normalize the data: subtract the mean image
    if normalize:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
#     X_train = X_train.transpose(0, 3, 1, 2).copy()
#     X_val = X_val.transpose(0, 3, 1, 2).copy()
#     X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def get_RICE_data() -> dict:
    """Load the rice dataset.

    Returns
        the train/val/test data and labels
    """
    df = pd.read_csv('./rice/riceClassification.csv')
    X=np.array(df.iloc[:,:-1])
    y=np.array(df.iloc[:,-1])

    X_train_RICE, X_test_RICE, y_train_RICE, y_test_RICE = train_test_split(X, y, test_size=0.2, random_state=44, shuffle =True)
    X_train_RICE, X_val_RICE, y_train_RICE, y_val_RICE = train_test_split(X_train_RICE, y_train_RICE, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    data = {
        "X_train": X_train_RICE,
        "y_train": y_train_RICE,
        "X_val": X_val_RICE,
        "y_val": y_val_RICE,
        "X_test": X_test_RICE,
        "y_test": y_test_RICE,
    }
    return data


def get_MUSHROOM_data(validation: float, testing: float = 0.2) -> dict:
    """Load the mushroom dataset.

    Parameters:
        validation: portion of the dataset used for validation
        testing: portion of the dataset used for testing

    Returns
        the train/val/test data and labels
    """
    X_train = np.load("mushroom/X_train.npy")
    y_train = np.load("mushroom/y_train.npy")
    y_test = np.load("mushroom/y_test.npy")
    X_test = np.load("mushroom/X_test.npy")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation / (1 - testing), random_state=123
    )
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    return data


def construct_MUSHROOM():
    """Convert raw categorical data from mushroom dataset to one-hot encodings."""
    dataset = pd.read_csv("mushroom/mushrooms.csv")
    y = dataset["class"]
    X = dataset.drop("class", axis=1)
    Encoder_X = LabelEncoder()
    for col in X.columns:
        X[col] = Encoder_X.fit_transform(X[col])
    Encoder_y = LabelEncoder()
    y = Encoder_y.fit_transform(y)
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    np.save("mushroom/X_train.npy", X_train)
    np.save("mushroom/y_train.npy", y_train)
    np.save("mushroom/X_test.npy", X_test)
    np.save("mushroom/y_test.npy", y_test)
