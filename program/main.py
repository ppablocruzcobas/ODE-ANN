# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


import sys
import window as w
import gc

from window_impls import WindowImpls
import keras.backend as K


if __name__ == "__main__":
    # Enable the Automatic Garbage Collector
    gc.enable()

    app = w.QtWidgets.QApplication(sys.argv)
    wnd = w.QtWidgets.QMainWindow()
    app.setApplicationDisplayName("RBNN")

    ui = w.Ui_MainWindow()
    ui.setupUi(wnd)
    wUI = WindowImpls(ui)

    ui.btnSolve.clicked.connect(wUI.onBtnSolveClick)
    ui.btnLoadFromFile.clicked.connect(wUI.onBtnLoadFromFileClick)
    ui.cbPoints.currentIndexChanged.connect(wUI.onCbPointsSelectionChange)

    ui.editEquation.textChanged.connect(wUI.onEditValueChange)
    ui.editInitialCondition.textChanged.connect(wUI.onEditValueChange)
    ui.editNeurons.textChanged.connect(wUI.onEditValueChange)
    ui.editSolutionInterval.textChanged.connect(wUI.onEditValueChange)
    ui.editAccuracy.textChanged.connect(wUI.onEditValueChange)
    ui.editIterations.textChanged.connect(wUI.onEditValueChange)

    ui.cbPlotSolution.stateChanged.connect(
        lambda: wUI.onPlotSelectionChange(0))
    ui.cbPlotLoss.stateChanged.connect(
        lambda: wUI.onPlotSelectionChange(1))

    wnd.show()
    sys.exit(app.exec_())
