# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


import numpy as np


class Linear:
    def __init__(self, *coefficients):
        self.coefficients = coefficients

    def __call__(self, x):
        return np.dot(x, self.coefficients)

    def fit_eval(self, x, *params):
        self.coefficients = params
        return self.__call__(x)


class Polynomial:
    def __init__(self, *coefficients):
        """ input: coefficients are in the form a_n, ...a_1, a_0
        """
        # for reasons of efficiency we save the coefficients in reverse order,
        # i.e. a_0, a_1, ... a_n
        self.coefficients = coefficients[::-1]
        # tuple is also turned into list

    def __call__(self, x):
        res = 0
        for index, coeff in enumerate(self.coefficients):
            res += coeff * x ** index
        return res

    def degree(self):
        return len(self.coefficients) - 1

    def fit_eval(self, x, *params):
        self.coefficients = params
        return self.__call__(x)
