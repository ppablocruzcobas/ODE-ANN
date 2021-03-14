# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


from tensorflow.keras.callbacks import Callback

import numpy as np


class EarlyStoppingByLoss(Callback):
    """
    My custom callback to stop training when loss reaches a `baseline` value.
    """
    def __init__(self, monitor='loss', baseline=0.01, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        """
        This method is called every time an epoch of training is finished.
        """
        current = logs.get(self.monitor)

        if current is None:
            warnings.warn("Early stopping requires %s available!" %
                          self.monitor, RuntimeWarning)

        if current < self.baseline:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class Progress(Callback):
    """
    My custom callback to monitor changes in `loss` values during training.
    """
    def __init__(self, monitor, epochs, progress, widget):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.epochs = epochs
        self.progress = progress
        self.widget = widget
        self.mloss = np.inf

    def on_epoch_end(self, epoch, logs={}):
        """
        This method is called every time an epoch of training is finished.
        """
        current = logs.get(self.monitor)
        if self.progress is not None:
            self.progress.setValue(100 * epoch / self.epochs)
        if self.widget is not None:
            if current < self.mloss:
                self.mloss = current
                self.widget.setText('   ' + self.monitor +
                                    ' : ' + '{:.4f}'.format(current))
