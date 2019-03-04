# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


from cauchyp import IVP
from Equation import Expression
from rbnn import RBNN
from PyQt5.QtWidgets import QFileDialog, QProgressBar, QLabel
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator

import time
import threading as thr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


class WindowImpls():
    """
    Implement the slots of some of the events that may be fired by the
    components declared in `ui` class.
    """
    def __init__(self, ui):
        """
        `ui` is the class autogenerated by pyuic5.
        """
        self.ui = ui
        self.__data = None
        self.__trained = False
        self.__progressBar = None
        self.__init_progress_bar()
        self.__hide_progress_bar()

        self.__validate_edit_entry()

    def __init_progress_bar(self):
        if self.__progressBar is None:
            self.__progressBar = QProgressBar()
            self.__progressBar.setMinimum(0)
            self.__progressBar.setMaximum(100)
            self.__progressBar.setValue(0)
            self.__progressBar.width()
            self.__progressBar.setVisible(True)
            self.__lbLoss = QLabel('   loss: 0.0')
            self.__lbLoss.setVisible(True)
            self.__rgSpacer = QLabel('   ')
            self.__rgSpacer.setVisible(True)
            self.ui.statusBar.addWidget(self.__lbLoss, 1)
            self.ui.statusBar.addWidget(self.__progressBar, 4)
            self.ui.statusBar.addWidget(self.__rgSpacer, 1)
        else:
            self.__lbLoss.setText('   loss : 0.0')
            self.__progressBar.setValue(0)
            self.__progressBar.setVisible(True)
            self.__lbLoss.setVisible(True)

    def __hide_progress_bar(self):
        self.__progressBar.setVisible(False)

    def __validate_edit_entry(self):
        # Regular Expression for `initial condition`
        __re = QRegExp(
            "^[+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+)([eE][-+]?[0-9]+)?[]?[,][]? [+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+)([eE][-+]?[0-9]+)?$")
        __validator = QRegExpValidator(__re)
        self.ui.editInitialCondition.setValidator(__validator)

        # Regular Expression for `int` representations
        __re = QRegExp("[1-9]+[0-9]+")
        __int_validator = QRegExpValidator(__re)
        self.ui.editIterations.setValidator(__int_validator)

        # Regular Expression for `float` representations
        __re = QRegExp(
            "^[+]?([0-9]+(\.[0-9]+)?|\.[0-9]+)([eE][-+]?[0-9]+)?$")
        __float_validator = QRegExpValidator(__re)
        self.ui.editAccuracy.setValidator(__float_validator)

    def __enable_widgets(self, value):
        self.ui.btnSolve.setEnabled(value)
        self.ui.editEquation.setEnabled(value)
        self.ui.editInitialCondition.setEnabled(value)
        if value is True and self.ui.cbPoints.currentIndex() == 1:
            self.ui.editSolutionInterval.setEnabled(value)
        elif value is False:
            self.ui.editSolutionInterval.setEnabled(value)
        self.ui.editNeurons.setEnabled(value)
        self.ui.editAccuracy.setEnabled(value)
        self.ui.editIterations.setEnabled(value)
        self.ui.cbActivationFunction.setEnabled(value)
        self.ui.cbOptimizer.setEnabled(value)
        self.ui.cbPlotLoss.setEnabled(value)
        self.ui.cbPlotSolution.setEnabled(value)
        self.ui.cbPoints.setEnabled(value)
        self.ui.btnLoadFromFile.setEnabled(value)

    def __exec_train(self):
        self.__enable_widgets(False)

    def onEditValueChange(self):
        """
        If any of the edits is empty then the button 'Solve' is disabled
        to avoid issues in execution.
        """
        self.ui.btnSolve.setEnabled(self.ui.editEquation.text() != "" and
                                    self.ui.editInitialCondition.text() != "" and
                                    self.ui.editNeurons.text() != "" and
                                    self.ui.editSolutionInterval.text() != "" and
                                    self.ui.editAccuracy.text() != "" and
                                    self.ui.editIterations.text() != "")

    def onCbPointsSelectionChange(self, index):
        self.ui.editSolutionInterval.setEnabled(index == 1)
        self.ui.editSolutionInterval.setClearButtonEnabled(index == 1)
        self.ui.editSolutionInterval.clear()
        self.ui.btnLoadFromFile.setVisible(index == 0)
        self.__data = None

    def onBtnLoadFromFileClick(self):
        """
        Instead of writing list of points on your own,
        just load it from a `.txt` file.
        """
        __dfile = QFileDialog.getOpenFileName(caption="Load data",
                                              directory=".",
                                              filter="Text files (*.txt)")
        if __dfile[0] is not '':
            try:
                self.__data = np.loadtxt(__dfile[0])
                # self.ui.editNeurons.setMaximum(len(self.__data))
                self.__data = self.__data.reshape(len(self.__data), 1)
                self.ui.editSolutionInterval.setText(__dfile[0])
            except:
                self.ui.statusBar.showMessage('Invalid file format', -1)

    def onPlotSelectionChange(self, index):
        if index == 1 and self.__trained:
            if self.ui.cbPlotLoss.isChecked():
                self.__fig_loss = plt.figure("Loss")
                sbn.set(font_scale=1)
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.plot(range(1, 1 + len(self.__h.history['loss'])),
                         self.__h.history['loss'], "r-", label='loss')
                plt.legend()
                plt.show(self.__fig_loss)
                plt.savefig("loss.png", format="png")
            else:
                try:
                    plt.close(self.__fig_loss)
                except:
                    pass
        elif index == 0 and self.__trained:
            if self.ui.cbPlotSolution.isChecked():
                self.__fig_result = plt.figure("Result")
                sbn.set(font_scale=1)
                plt.xlabel("t")
                plt.ylabel("x")
                plt.scatter(self.__interval, self.__values, marker="*",
                            label='results')
                plt.legend()
                plt.show(self.__fig_result)
                plt.savefig("result.png", format="png")
            else:
                try:
                    plt.close(self.__fig_result)
                except:
                    pass

    def onBtnSolveClick(self):
        """
        Solve the IVP and plot solution (in case of checked).
        The graphic a results are saved by default, in files `result.png`
        and `result.txt`, respectly.
        """
        __condition = eval(self.ui.editInitialCondition.text())

        if self.__data is None:
            try:
                self.__interval = np.array(
                    eval(self.ui.editSolutionInterval.text()))
                self.__interval = self.__interval.reshape(len(self.__interval), 1)
            except:
                self.ui.statusBar.showMessage('Invalid solution \
                interval format', -1)
                return
        else:
            self.__interval = self.__data

        self.__trained = False
        # self.ui.editNeurons.setMaximum(len(self.__interval))

        try:
            ivp = IVP(Expression(self.ui.editEquation.text(), ['t', 'x']),
                      __condition[0], __condition[1])
        except:
            self.ui.statusBar.showMessage('Invalid equation type', -1)
            return

        network = RBNN(int(self.ui.editNeurons.text()),
                       self.ui.cbActivationFunction.currentText(),
                       self.ui.cbOptimizer.currentText(),
                       int(self.ui.editIterations.text()),
                       float(self.ui.editAccuracy.text()))

        __thr = thr.Thread(target=self.__exec_train)
        __thr.start()

        __start_t = time.time()
        self.__init_progress_bar()
        self.__h = network.solve_ivp(ivp, self.__interval,
                                     self.__progressBar, self.__lbLoss)
        self.__hide_progress_bar()
        __end_t = time.time()

        self.__lbLoss.setText(
            self.__lbLoss.text() + '. The training took '
            + '{:.2f}'.format(__end_t - __start_t) + ' segs.')

        try:
            self.__values = ivp(self.__interval)
            np.savetxt("result.txt", self.__values, fmt='%10.4f')
        except:
            self.ui.statusBar.showMessage('An error ocurred trying to \
            save data. Check your code!', -1)
            return

        self.__trained = True
        self.ui.editNeurons.setMaximum(999999)

        self.onPlotSelectionChange(0)
        self.onPlotSelectionChange(1)

        self.__enable_widgets(True)
