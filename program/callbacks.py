# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


from keras.callbacks import Callback

import keras.backend as K


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


class LRScheduler(Callback):
    def __init__(self, min_lr=0.00001, max_lr=0.01, epochs=1000, verbose=1):
        super(Callback, self).__init__()
        self.__min_lr = min_lr
        self.__max_lr = max_lr
        self.__epochs = epochs
        self.__verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        __x = epoch / self.__epochs
        __lr = self.__max_lr - (self.__max_lr - self.__min_lr) * __x
        K.set_value(self.model.optimizer.lr, __lr)

        if self.__verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, __lr))


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

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.__monitor)
        if self.__progress is not None:
            self.__progress.setValue(100 * epoch / self.__epochs)
        if self.__lb_loss is not None:
            self.__lb_loss.setText('   ' + self.__monitor +
                                   ' : ' + '{:.2f}'.format(current))
