
author = 'Pedro Pablo'


from nnet import IVP, NNet
from nnet import Model
from Equation import Expression
from PyQt5.QtWidgets import QFileDialog
from ui.main_window_utils import *
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sbn


class Ui_MainWindowImplementations():
    """
    Implement the slots of some of the events that may be fired by the
    components declared in `ui` class.
    """
    def __init__(self, ui):
        """
        `ui` is the class autogenerated by pyuic5.
        """
        self.ui = ui
        self.data = None

        self.progress_bar, self.lb_loss = update_status_bar(self.ui.statusBar,
                                                            None, None)
        self.widgets = [self.ui.cbPoints, self.ui.editSolutionInterval,
                        self.ui.btnSolve,
                        self.ui.editEquation, self.ui.cbModel,
                        self.ui.editInitialCondition, self.ui.editNeurons,
                        self.ui.editAccuracy, self.ui.editIterations,
                        self.ui.cbActivationFunction, self.ui.cbOptimizer,
                        self.ui.cbPlotLoss, self.ui.cbPlotSolution,
                        self.ui.cbTensorBoard,
                        self.ui.btnLoadFromFile]

        self.path= 'results/'
        # Create path `results` if doesn't exists
        os.makedirs(self.path, exist_ok=True)

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

    def onCbModelSelectionChange(self, index):
        self.ui.cbActivationFunction.setEnabled(index not in [0, 1, 3])
        self.ui.lbActivationFunction.setEnabled(index not in [0, 1, 3])
        self.ui.editNeurons.setEnabled(index not in [0, 1])
        self.ui.lbNeurons.setEnabled(index not in [0, 1])

    def onCbPointsSelectionChange(self, index):
        self.ui.editSolutionInterval.setEnabled(index == 1)
        self.ui.editSolutionInterval.setClearButtonEnabled(index == 1)
        self.ui.editSolutionInterval.clear()
        self.ui.btnLoadFromFile.setVisible(index == 0)
        self.data = None

    def onBtnLoadFromFileClick(self):
        """
        Instead of writing list of points on your own,
        just load it from a `.txt` file.
        """
        dfile = QFileDialog.getOpenFileName(caption="Load data",
                                              directory=".",
                                              filter="Text files (*.txt)")
        if dfile[0] is not '':
            try:
                self.data = np.loadtxt(dfile[0])
                self.data = self.data.reshape(len(self.data), 1)
                self.ui.editSolutionInterval.setText(dfile[0])
            except:
                message(self.ui.statusBar, 'Invalid file format')

    def check_fo_saving_graph(self):
        plt.figure("Loss")
        sbn.set(font_scale=1)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(range(1, 1 + len(self.h.history['loss'])),
                    self.h.history['loss'], "r-", label='loss')
        plt.legend()
        plt.savefig(self.path + "loss.png", format="png")
        if self.ui.cbPlotLoss.isChecked():
            plt.show()
            
        plt.figure("Result")
        sbn.set(font_scale=1)
        plt.xlabel("x")
        plt.ylabel("y")
        x = np.array(np.linspace(
            self.interval[0], self.interval[-1], 100))
        x = x.reshape(len(x), 1)
        plt.plot(x, self.network(x), '-g', label='NNet')
        plt.legend()
        plt.savefig(self.path + "result.png", format="png")
        if self.ui.cbPlotSolution.isChecked():
            plt.show()

    def onBtnSolveClick(self):
        """
        Solve the IVP and plot solution (in case of checked).
        """
        condition = eval(self.ui.editInitialCondition.text())

        if self.data is None:
            try:
                self.interval = np.array(
                    eval(self.ui.editSolutionInterval.text()))
                self.interval = self.interval.reshape(len(self.interval), 1)
            except:
                message(self.ui.statusBar, 'Invalid solution interval format')
                return
        else:
            self.interval = self.data

        try:
            ivp = IVP(self.ui.editEquation.text(),
                      condition[0], condition[1])
        except:
            message(self.ui.statusBar, 'Invalid equation type')
            return

        try:
            model = Model(self.ui.cbModel.currentText(),
                          int(self.ui.editNeurons.text()),
                          self.ui.cbActivationFunction.currentText())
            self.network = NNet(model.get(),
                                self.ui.cbOptimizer.currentText(),
                                int(self.ui.editIterations.text()),
                                float(self.ui.editAccuracy.text()),
                                self.ui.cbTensorBoard.isChecked())
        except:
            message(self.ui.statusBar,
                    'An error has occured creating the model')
            return

        self.network.set_updatable_widgets(self.progress_bar, self.lb_loss)

        enable_widgets(self.widgets, False)

        # Calculates the time of the training session
        start_t=time.time()
        update_status_bar(self.ui.statusBar, self.progress_bar, self.lb_loss)
        self.h=self.network.solve_ivp(ivp, self.interval)
        hide([self.progress_bar])
        end_t=time.time()

        # Shows the calculated time
        self.lb_loss.setText(
            self.lb_loss.text() + '. The training took '
            + '{:.2f}'.format(end_t - start_t) + ' segs.')

        try:
            self.values = self.network(self.interval)
            np.savetxt(self.path + "result.txt", self.values, fmt='%10.4f')
        except:
            message(self.ui.statusBar,
                    'An error ocurred trying to save data!')
            return

        self.check_fo_saving_graph()

        enable_widgets(self.widgets, True)
        self.onCbModelSelectionChange(self.ui.cbModel.currentIndex())
