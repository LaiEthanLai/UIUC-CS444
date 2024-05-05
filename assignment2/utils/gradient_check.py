# Credit: http://cs231n.github.io/assignments2018/assignment1/

from typing import Callable

import numpy as np


def eval_numerical_gradient(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    verbose: bool = True,
    h: float = 0.00001,
):
    """A naive implementation of numerical gradient of f at x

    Parameters:
        f: a function that takes a single argument
        x: the point to evaluate the gradient at

    Returns:
        the numerical gradient
    """
    f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension
    # print('grad shape: {}'.format(grad.shape))
    return grad
