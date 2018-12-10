# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


import numpy as np


class Linear:
    """
    A class implementing a `linear form` needed operations
    to initialize weights.
    """
    def __init__(self, *coefficients):
        self.__coeffs = coefficients

    def __repr__(self):
        """
        Method to return the canonical string representation
        of a `linear form`.
        """
        return "Linear" + str(self.__coeffs)

    def __call__(self, x):
        """
        Returns the `dot product` of `x` and `coefficients`.
        """
        return np.dot(x, self.__coeffs)

    def fit_eval(self, x, *params):
        """
        Only to be used with `curve_fit` from `scipy.optimize`.
        """
        self.__coeffs = params
        return self.__call__(x)


class Polynomial:
    """
    A class implementing the polynomials needed operations
    to initialize weights.
    """
    def __init__(self, *coefficients):
        """
        `coefficients` = coefficients are in the form aN, ...a1, a0
        """
        # For reasons of efficiency we save the coefficients in reverse order,
        # i.e. a0, a1, ... aN
        self.__coeffs = coefficients[::-1]
        # Also `tuple` is turned into `list`.

    def __repr__(self):
        """
        Method to return the canonical string representation
        of a polynomial.
        """
        # The internal representation is in reverse order,
        # so we have to reverse the list
        return "Polynomial" + str(self.__coeffs[::-1])

    def __call__(self, x):
        """
        Returns the polynomial evaluated at point `x` (may be a list).
        """
        __res = 0
        for __index, __coeff in enumerate(self.__coeffs):
            __res += __coeff * np.power(x, __index)
        return __res

    def degree(self):
        """
        Returns the polynomial degree.
        """
        return len(self.__coeffs) - 1

    def get_coeffs(self):
        """
        Returns the polynomial coefficients.
        """
        return self.__coeffs

    def fit_eval(self, x, *params):
        """
        Only to be used with `curve_fit` from `scipy.optimize`.
        """
        self.__coeffs = params
        return self.__call__(x)

    def derivative(self):
        """
        Returns the derivative of the polynomial.
        """
        __d_coeffs = []
        __exp = 1
        for i in range(1, len(self.__coeffs)):
            __d_coeffs.append(self.__coeffs[i] * __exp)
            __exp += 1
        return Polynomial(*__d_coeffs[::-1])
