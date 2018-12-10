# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.layers import Dense
from keras.utils import plot_model
from initializers import PolynomialInitializer, LinearInitializer

import keras.backend as K
import callbacks as C


class RBNN():
    """
    Class that make the training to solve the IVP
    """
    def __init__(self, units, afunc, optimizer, epochs, acc):
        """
        Initialize the class main elements.

        `units` = number of neurons in the hidden layer.
        `afunc` = activation function to be used (sigmoid, tanh, relu).
        `optimizer` = which method of optimization to use:
        `rmsprop`, `adam`, `sgd`
        `epochs` = number of times that the process will be repeated.
        `acc` = desired threshold for the solutions.
        """
        self.__units = units
        self.__afunc = afunc
        self.__optimizer = optimizer
        self.__epochs = epochs
        self.__acc = acc
        # Used to decide if show (1) or not (0) the training progress.
        self.__verbose = 1

    def __gradient(self):
        """
        Computes the gradient of the network outputs with respect
        to the network inputs
        """
        return K.gradients(self.__model.outputs, self.__model.inputs)

    def __trial_solution(self, t, _eval=False):
        """
        Returns the value of the proposed solution at the point `t`.
        """
        __t = K.tf.convert_to_tensor(t, dtype=K.tf.float32, name='t')
        __result = self.__x0 + (__t - self.__t0) * self.__model(__t)

        if _eval:
            return K.eval(__result)
        else:
            return __result

    def __build_loss_function(self, t):
        """
        Constructs the loss function to be used update the weights.
        `t` is the tensor of the points in which the equation want to be solved.
        """
        __t = K.tf.convert_to_tensor(t, dtype=K.tf.float32, name='points')

        def error(tensor_in, tensor_out):
            __d = self.__model(__t) + (__t - self.__t0) * self.__gradient()
            __f = self.__eqfunc(__t, self.__trial_solution(__t))
            __diff = __d - __f
            return 0.5 * K.sum(K.square(__diff))

        return error

    def solve_ivp(self, ivp, points, progress=None, lb_loss=None):
        self.__eqfunc = ivp.get_equation()
        self.__t0 = K.eval(K.tf.convert_to_tensor(
            [[ivp.get_t0()]], dtype=K.tf.float32, name='t0'))
        self.__x0 = K.eval(K.tf.convert_to_tensor(
            [[ivp.get_x0()]], dtype=K.tf.float32, name='x0'))

        __checkpoint = ModelCheckpoint('model.h5', monitor='loss',
                                       verbose=self.__verbose,
                                       save_best_only=True, mode='min')
        __NaN = TerminateOnNaN()
        # Custom Callback to stop when 'baseline' is reached
        __earlystop = C.EarlyStoppingByLoss(monitor='loss',
                                          baseline=self.__acc,
                                          verbose=self.__verbose)
        # __scheduler = C.LRScheduler(K, min_lr=0.00001, max_lr=0.001,
                                    # epochs=self.__epochs,
                                    # verbose=self.__verbose)
        # Show progress (also custom callback)...
        __progress = C.ShowProgress('loss', self.__epochs, progress, lb_loss)
        # Custom loss function
        __loss_f = self.__build_loss_function(points)

        try:
            __points = list(points.reshape(1, len(points))[0])
            # If the initial condition is not in the interval, include it.
            if ivp.get_x0() not in __points:
                __points.insert(0, ivp.get_x0())
            __values = []
            __values.append(ivp.get_x0())
            for i in range(len(__points) - 1):
                __values.append(0)

            __polynomial = PolynomialInitializer(self.__eqfunc, self.__units,
                                                 __points, __values)
            __linear = LinearInitializer(K, self.__afunc,
                                         __points, __polynomial.get_weights())
        except:
            __polynomial = 'glorot_uniform'
            __linear = 'glorot_uniform'

        self.__model = Sequential()
        self.__model.add(Dense(1, input_shape=(1,),
                               name='input_layer'))
        self.__model.add(Dense(self.__units, activation=self.__afunc,
                               kernel_initializer=__polynomial,
                               name='hidden_layer'))
        self.__model.add(Dense(1, kernel_initializer=__linear,
                               name='output_layer'))

        if self.__verbose:
            self.__model.summary()

        self.__model.compile(loss=__loss_f, optimizer=self.__optimizer)
        # Save the model diagram
        plot_model(self.__model, to_file='model.png', show_shapes=True,
                   show_layer_names=True, rankdir='TB')

        __data_tensor = K.eval(
            K.tf.convert_to_tensor(points, name='data_tensor'))
        __history = self.__model.fit(__data_tensor, __data_tensor,
                                     batch_size=len(__data_tensor),
                                     epochs=self.__epochs,
                                     verbose=self.__verbose,
                                     callbacks=[__checkpoint, __earlystop,
                                                __NaN, __progress])

        # Load the `optimal` weights
        self.__model.load_weights('model.h5', by_name=True)

        ivp.set_solution(self.__trial_solution)

        return __history
