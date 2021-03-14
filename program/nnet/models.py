
__author__ = 'Pedro Pablo'


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from Equation import Expression
from nnet.layers import ExpandedLayer, RBFLayer


def create_expansion(name='ChNN', order=7):
    """
    Returns the (Chebyshev, Legendre) expansion of order `order`.
    
    `name` = type of expansion to be created.
    `order` = order of the expansion created
              (at this moment, order > 7 is giving problems with Tensorflow!!!)
    """
    x = Expression('x', ['x'])
    T = [Expression('1', ['x']), x]
    if name == 'ChNN':
        for n in range(2, order):
            T.append(2 * x * T[n - 1] - T[n - 2])
    elif name == 'LeNN':
        for n in range(2, order):
            T.append(((2 * n - 1) * x * T[n - 1]
                           - (n - 1) * T[n - 2]) / n)
    else:
        raise Exception('Unknown expansion type!')
    return T


class Model():
    """
    Class that creates the model to be used in the NNet.
    
    `name` = type of model to create:
             RBF - Radial Basis Functions
             MLP - Multilayer Perceptron (1 hidden leyer only)
             ChNN - Functional Link Chebyshev Neural Network
             LeNN - Functional Link Legendre Neural Network
             HeNN - Functional Link Hermite Neural Network
    `units` = neurons in the hidden layer
              (Used only in RBF and MLP)
    `activation` = activation function
                   (Used only in RBF and MLP)
                   (The activation functionin the other cases is `tanh`)
    """
    def __init__(self, name='RBF', units=10, activation='sigmoid'):
        self.model = Sequential()
        if name == 'RBF':
            rbflayer = RBFLayer(units, input_shape=(1, ))
            self.model.add(rbflayer)
            self.model.add(Dense(1))
        elif name == 'MLP':
            self.model.add(Dense(1, input_shape=(1, ), use_bias=False,
                            name='input_layer'))
            self.model.add(Dense(units, activation=activation,
                             kernel_initializer='glorot_uniform',
                             name='hidden_layer'))
            self.model.add(Dense(1, kernel_initializer='glorot_uniform',
                             use_bias=False,
                             name='output_layer'))
        elif name == 'ChNN' or name == 'LeNN':
            T = create_expansion(name)
            self.model.add(ExpandedLayer(1, T, input_shape=(1, ),
                             name='expansion_block'))
        else:
            raise Exception("Unknown model name!")

    def get(self):
        return self.model
