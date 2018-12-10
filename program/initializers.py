# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


from scipy.optimize import curve_fit
from utils import Linear, Polynomial
from keras.initializers import Initializer

import numpy as np


class PolynomialInitializer(Initializer):
    """
    Initialize hidden layer weights using orthogonal polynomials.
    `f` = right part of the ODE
    `degree` = polynomial degree (number of units in hidden layer)
    `t` = times interval (first element equals `t0`)
    `x` = first element equals `x0` (the rest is `0`)
    """
    def __init__(self, f, degree, t, x):
        __coeffs = degree * [1.0]
        self.__f = f
        self.__op = Polynomial(__coeffs)
        self.__dp = self.__op.derivative()
        self.__res = curve_fit(self.__fit_eval, t, x, __coeffs)[0]

    def __call__(self, shape, dtype=None):
        """
        Returns the tensor to be passed to the layer
        """
        self.__res = np.array(self.__res)
        return self.__res.reshape(1, len(self.__res))

    def __fit_eval(self, t, *params):
        return self.__dp.fit_eval(t, *params) - self.__f(t, self.__op.fit_eval(t, *params))

    def get_weights(self):
        """
        Returns the learned weights as a list.
        Not to be used in the model, because the model only accept tensors
        as input.
        This is for testing purposes only.
        """
        return self.__res


class LinearInitializer(Initializer):
    """
    Initialize output layer weights using multivariate linear regression.
    `K` = Keras backend already loaded (this execution saves time)
    `afunc` = activation function in use
    `t` = times interval (first element equals `t0`)
    `w` = weights returned from the `PolynomialInitializer`
    """
    def __init__(self, K, afunc, t, w):
        __coeffs = len(w) * [1]
        __lin = Linear(__coeffs)
        __p = Polynomial(__coeffs)
        __x = __p.fit_eval(t, *w)

        __vector = []
        for __t in t:
            __temp = []
            for j in range(len(w)):
                if afunc == 'sigmoid':
                    __temp.append(K.eval(K.sigmoid(__t * w[j])))
                elif afunc == 'tanh':
                    __temp.append(K.eval(K.tanh(__t * w[j])))
                elif afunc == 'relu':
                    __temp.append(K.eval(K.relu(__t * w[j])))
            if __temp is not []:
                __vector.append(__temp)

        self.__res = curve_fit(__lin.fit_eval, __vector, __x, __coeffs)[0]

    def __call__(self, shape, dtype=None):
        """
        Returns the tensor to be passed to the layer
        """
        self.__res = np.array(self.__res)
        return self.__res.reshape(len(self.__res), 1)

    def get_weights(self):
        """
        Returns the learned weights as a list.
        Not to be used in the model, because the model only accept tensors
        as input.
        This is for testing purposes only.
        """
        return self.__res
