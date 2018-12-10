# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


from keras.callbacks import Callback

import numpy as np


class EarlyStoppingByLoss(Callback):
    """
    My custom callback to stop training when loss reaches a `baseline` value.
    """
    def __init__(self, monitor='loss', baseline=0.01, verbose=1):
        super(Callback, self).__init__()
        self.__monitor = monitor
        self.__baseline = baseline
        self.__verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.__monitor)

        if current is None:
            warnings.warn("Early stopping requires %s available!" %
                          self.monitor, RuntimeWarning)

        if current < self.__baseline:
            if self.__verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class ShowProgress(Callback):
    """
    My custom callback to stop training when loss reaches a `baseline` value.
    """
    def __init__(self, monitor, epochs, progress, lb_loss):
        super(Callback, self).__init__()
        self.__monitor = monitor
        self.__epochs = epochs
        self.__progress = progress
        self.__lb_loss = lb_loss
        self.__min_loss = np.inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.__monitor)
        if self.__progress is not None:
            self.__progress.setValue(100 * epoch / self.__epochs)
        if self.__lb_loss is not None:
            if current < self.__min_loss:
                self.__min_loss = current
            self.__lb_loss.setText('   ' + self.__monitor +
                                   ' : ' + '{:.2f}'.format(self.__min_loss))
