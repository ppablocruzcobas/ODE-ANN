
__author__ = 'Pedro Pablo'

import os
import tensorflow.keras.backend as K
import tensorflow as tf
import scipy
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN
from nnet.callbacks import EarlyStoppingByLoss, Progress


class NNet:
    def __init__(self, model, optimizer='RMSprop',
                 epochs=2500, accuracy=1e-2,
                 tensorboard=True, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.accuracy = accuracy
        self.verbose = verbose
        self.tensorboard = tensorboard
        
        # Create path `results` if doesn't exists
        self.path = 'results/'
        os.makedirs(self.path, exist_ok=True)

    def __call__(self, x):
        return K.eval(self.solution(x))

    def set_updatable_widgets(self, progress, label):
        """
        Set widgets that will monitor for loss and progress of training session
        `progress` = progress bar that is updated at every epoch of training
        `label` = label that show the minimum `loss` at current epoch
        """
        self.progress = progress
        self.label = label

    def gradient(self):
        """
        Computes the gradient of the network outputs with respect
        to the network inputs
        """
        return K.gradients(self.model.outputs, self.model.inputs)

    def loss(self, x0, x_tensor, y_tensor=None):
        """
        Constructs the loss function to be used to update the weights.
        `x_tensor` = tensor of points at which the equation
                     want to be solved.
        """
        def error(tensor_in, tensor_out):
            d = self.model(x_tensor) + (x_tensor - x0) * self.gradient()
            if y_tensor == None:
                f = self.function(x_tensor, self.solution(x_tensor))
            else:
                f = self.function(x_tensor, y_tensor)
            return K.sum(K.pow(d - f, 2))
        return error

    def euler_solve(self, ivp, x):
        return None
        # Converts `x` to a one-dimensional list
        # x = list(x.reshape(1, len(x))[0])
        # try:
            #  Here should be implemented the Euler's method
            # result = scipy.integrate.solve_ivp(ivp.function(), [ivp.x0(), x[-1]],
            #                                   [ivp.y0()], t_eval=x)
            # y = result.y[0]
        # except:
        #    return None
        # y = y.reshape(len(y), 1)
        # return K.tf.convert_to_tensor(y, dtype=K.tf.float32, name='y_tensor')

    def function(self, x, y):
        return eval(self.func, {'K': K, 'x': x, 'y': y})

    def solution(self, x):
        """
        Returns the value of the proposed solution at the point `x`.
        Proposed solution is:
                       y(x) = y0 + (x-x0) * N(x, p)
        where y(x0) = y0 is the initial condition and N(x, p) is the NNet.
        """
        x_tensor = K.tf.convert_to_tensor(
            x, dtype=K.tf.float32, name='x_tensor')
        return self.y0 + (x_tensor - self.x0) * self.model(x_tensor)

    def solve_ivp(self, ivp, x):
        if self.tensorboard == True:
            # Write graph data to file to watch it in TensorBoard
            writer = tf.summary.FileWriter('logs/',
                                           graph=tf.get_default_graph())
        # Convert data to tensor
        x0 = K.convert_to_tensor(
            [[ivp.x0()]], dtype=K.tf.float32, name='x0')
        y0 = K.convert_to_tensor(
            [[ivp.y0()]], dtype=K.tf.float32, name='y0')
        x_tensor = K.convert_to_tensor(x, dtype=K.tf.float32,
                                          name='x_tensor')
        # Find approximate solutions using Euler's method
        y_tensor = self.euler_solve(ivp, x)

        # Save it to be globally accessible
        self.func = ivp.convert_to_keras_expression()
        self.x0 = x0
        self.y0 = y0

        # Checkpoints to be call at every epoch of traning session
        checkpoint = ModelCheckpoint(self.path + 'weights.h5', monitor='loss',
                                     verbose=self.verbose,
                                     save_best_only=True, mode='min')
        NaN = TerminateOnNaN()

        # Custom checkpoints
        progress = Progress('loss', self.epochs, self.progress, self.label)
        early = EarlyStoppingByLoss(baseline=self.accuracy)

        if y_tensor != None:
            self.model.compile(loss=self.loss(x0, x_tensor, y_tensor),
                               optimizer=self.optimizer)
            history = self.model.fit(K.eval(x_tensor), K.eval(y_tensor),
                                     batch_size=len(x),
                                     epochs=self.epochs,
                                     verbose=self.verbose,
                                     callbacks=[checkpoint, NaN,
                                                progress, early])
            # Load the `optimal` weights
            self.model.load_weights(self.path + 'weights.h5')
            
        self.model.compile(loss=self.loss(x0, x_tensor),
                           optimizer=self.optimizer)
        
        # Save the model diagram (horizontal plot)
        plot_model(self.model, to_file=self.path + 'model.png',
                   show_shapes=True, show_layer_names=True, rankdir='LR')

        # Shows a summary of the model
        if self.verbose:
            self.model.summary()

        # Stores the history of the training session for future use, like
        #  plotting loss vs. epoch
        history=self.model.fit(K.eval(x_tensor), K.eval(x_tensor),
                                 batch_size=len(x),
                                 epochs=self.epochs,
                                 verbose=self.verbose,
                                 callbacks=[checkpoint, NaN,
                                            progress, early])
        # Load the `optimal` weights
        self.model.load_weights(self.path + 'weights.h5')

        return history
