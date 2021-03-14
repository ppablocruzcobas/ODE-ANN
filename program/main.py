
__author__ = 'Pedro Pablo'


import ui.main_window as w
from ui.main_window_implementations import Ui_MainWindowImplementations
from ui.main_window_utils import *


if __name__ == "__main__":
    import sys

    app = w.QtWidgets.QApplication(sys.argv)
    wnd = w.QtWidgets.QMainWindow()
    app.setApplicationDisplayName('ODENet')

    ui = w.Ui_MainWindow()
    ui.setupUi(wnd)
    wUI = Ui_MainWindowImplementations(ui)

    ui.btnSolve.clicked.connect(wUI.onBtnSolveClick)
    ui.btnLoadFromFile.clicked.connect(wUI.onBtnLoadFromFileClick)
    ui.cbPoints.currentIndexChanged.connect(wUI.onCbPointsSelectionChange)
    ui.cbModel.currentIndexChanged.connect(wUI.onCbModelSelectionChange)

    ui.editEquation.textChanged.connect(wUI.onEditValueChange)
    ui.editInitialCondition.textChanged.connect(wUI.onEditValueChange)
    ui.editSolutionInterval.textChanged.connect(wUI.onEditValueChange)
    ui.editAccuracy.textChanged.connect(wUI.onEditValueChange)

    ui.editInitialCondition.setValidator(initial_condition_validator())
    ui.editAccuracy.setValidator(float_validator())

    wnd.show()
    sys.exit(app.exec_())

