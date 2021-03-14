# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        MainWindow.setEnabled(True)
        MainWindow.resize(500, 428)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(500, 428))
        MainWindow.setMaximumSize(QtCore.QSize(500, 428))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icon.jpeg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setMinimumSize(QtCore.QSize(500, 406))
        self.centralWidget.setMaximumSize(QtCore.QSize(500, 406))
        self.centralWidget.setStyleSheet("")
        self.centralWidget.setObjectName("centralWidget")
        self.frIVP = QtWidgets.QFrame(self.centralWidget)
        self.frIVP.setGeometry(QtCore.QRect(10, 10, 482, 132))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frIVP.sizePolicy().hasHeightForWidth())
        self.frIVP.setSizePolicy(sizePolicy)
        self.frIVP.setMinimumSize(QtCore.QSize(482, 132))
        self.frIVP.setMaximumSize(QtCore.QSize(482, 132))
        self.frIVP.setFrameShape(QtWidgets.QFrame.Box)
        self.frIVP.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frIVP.setObjectName("frIVP")
        self.layoutWidget = QtWidgets.QWidget(self.frIVP)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 461, 32))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lbEquation = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbEquation.sizePolicy().hasHeightForWidth())
        self.lbEquation.setSizePolicy(sizePolicy)
        self.lbEquation.setMinimumSize(QtCore.QSize(80, 30))
        self.lbEquation.setMaximumSize(QtCore.QSize(80, 30))
        self.lbEquation.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbEquation.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lbEquation.setTextFormat(QtCore.Qt.RichText)
        self.lbEquation.setScaledContents(False)
        self.lbEquation.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lbEquation.setObjectName("lbEquation")
        self.horizontalLayout.addWidget(self.lbEquation)
        self.editEquation = QtWidgets.QLineEdit(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editEquation.sizePolicy().hasHeightForWidth())
        self.editEquation.setSizePolicy(sizePolicy)
        self.editEquation.setMinimumSize(QtCore.QSize(370, 30))
        self.editEquation.setMaximumSize(QtCore.QSize(370, 30))
        self.editEquation.setStatusTip("")
        self.editEquation.setInputMask("")
        self.editEquation.setText("")
        self.editEquation.setClearButtonEnabled(True)
        self.editEquation.setObjectName("editEquation")
        self.horizontalLayout.addWidget(self.editEquation)
        self.layoutWidget1 = QtWidgets.QWidget(self.frIVP)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 50, 168, 32))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lbInitialCondition = QtWidgets.QLabel(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbInitialCondition.sizePolicy().hasHeightForWidth())
        self.lbInitialCondition.setSizePolicy(sizePolicy)
        self.lbInitialCondition.setMinimumSize(QtCore.QSize(80, 30))
        self.lbInitialCondition.setMaximumSize(QtCore.QSize(80, 30))
        self.lbInitialCondition.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbInitialCondition.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lbInitialCondition.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lbInitialCondition.setObjectName("lbInitialCondition")
        self.horizontalLayout_2.addWidget(self.lbInitialCondition)
        self.editInitialCondition = QtWidgets.QLineEdit(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editInitialCondition.sizePolicy().hasHeightForWidth())
        self.editInitialCondition.setSizePolicy(sizePolicy)
        self.editInitialCondition.setMinimumSize(QtCore.QSize(80, 30))
        self.editInitialCondition.setMaximumSize(QtCore.QSize(80, 30))
        self.editInitialCondition.setStatusTip("")
        self.editInitialCondition.setInputMethodHints(QtCore.Qt.ImhExclusiveInputMask)
        self.editInitialCondition.setClearButtonEnabled(True)
        self.editInitialCondition.setObjectName("editInitialCondition")
        self.horizontalLayout_2.addWidget(self.editInitialCondition)
        self.layoutWidget2 = QtWidgets.QWidget(self.frIVP)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 90, 461, 32))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lbSolutionInterval = QtWidgets.QLabel(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbSolutionInterval.sizePolicy().hasHeightForWidth())
        self.lbSolutionInterval.setSizePolicy(sizePolicy)
        self.lbSolutionInterval.setMinimumSize(QtCore.QSize(80, 30))
        self.lbSolutionInterval.setMaximumSize(QtCore.QSize(80, 30))
        self.lbSolutionInterval.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lbSolutionInterval.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbSolutionInterval.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lbSolutionInterval.setObjectName("lbSolutionInterval")
        self.horizontalLayout_3.addWidget(self.lbSolutionInterval)
        self.cbPoints = QtWidgets.QComboBox(self.layoutWidget2)
        self.cbPoints.setMinimumSize(QtCore.QSize(80, 30))
        self.cbPoints.setMaximumSize(QtCore.QSize(80, 30))
        self.cbPoints.setObjectName("cbPoints")
        self.cbPoints.addItem("")
        self.cbPoints.addItem("")
        self.horizontalLayout_3.addWidget(self.cbPoints)
        self.editSolutionInterval = QtWidgets.QLineEdit(self.layoutWidget2)
        self.editSolutionInterval.setEnabled(False)
        self.editSolutionInterval.setMinimumSize(QtCore.QSize(0, 30))
        self.editSolutionInterval.setMaximumSize(QtCore.QSize(16777215, 30))
        self.editSolutionInterval.setToolTip("")
        self.editSolutionInterval.setStatusTip("")
        self.editSolutionInterval.setClearButtonEnabled(False)
        self.editSolutionInterval.setObjectName("editSolutionInterval")
        self.horizontalLayout_3.addWidget(self.editSolutionInterval)
        self.btnLoadFromFile = QtWidgets.QToolButton(self.layoutWidget2)
        self.btnLoadFromFile.setMinimumSize(QtCore.QSize(30, 30))
        self.btnLoadFromFile.setMaximumSize(QtCore.QSize(30, 30))
        self.btnLoadFromFile.setCheckable(False)
        self.btnLoadFromFile.setChecked(False)
        self.btnLoadFromFile.setAutoRepeat(False)
        self.btnLoadFromFile.setObjectName("btnLoadFromFile")
        self.horizontalLayout_3.addWidget(self.btnLoadFromFile)
        self.frRBNN = QtWidgets.QFrame(self.centralWidget)
        self.frRBNN.setGeometry(QtCore.QRect(30, 150, 440, 132))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frRBNN.sizePolicy().hasHeightForWidth())
        self.frRBNN.setSizePolicy(sizePolicy)
        self.frRBNN.setMinimumSize(QtCore.QSize(440, 132))
        self.frRBNN.setMaximumSize(QtCore.QSize(440, 132))
        self.frRBNN.setFrameShape(QtWidgets.QFrame.Box)
        self.frRBNN.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frRBNN.setObjectName("frRBNN")
        self.layoutWidget3 = QtWidgets.QWidget(self.frRBNN)
        self.layoutWidget3.setGeometry(QtCore.QRect(10, 10, 420, 111))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget3)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.lbModel = QtWidgets.QLabel(self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbModel.sizePolicy().hasHeightForWidth())
        self.lbModel.setSizePolicy(sizePolicy)
        self.lbModel.setMinimumSize(QtCore.QSize(80, 30))
        self.lbModel.setMaximumSize(QtCore.QSize(80, 30))
        self.lbModel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lbModel.setObjectName("lbModel")
        self.gridLayout.addWidget(self.lbModel, 0, 0, 1, 1)
        self.cbModel = QtWidgets.QComboBox(self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbModel.sizePolicy().hasHeightForWidth())
        self.cbModel.setSizePolicy(sizePolicy)
        self.cbModel.setMinimumSize(QtCore.QSize(94, 30))
        self.cbModel.setMaximumSize(QtCore.QSize(94, 30))
        self.cbModel.setStatusTip("")
        self.cbModel.setObjectName("cbModel")
        self.cbModel.addItem("")
        self.cbModel.addItem("")
        self.cbModel.addItem("")
        self.cbModel.addItem("")
        self.gridLayout.addWidget(self.cbModel, 0, 1, 1, 1)
        self.lbIterations = QtWidgets.QLabel(self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbIterations.sizePolicy().hasHeightForWidth())
        self.lbIterations.setSizePolicy(sizePolicy)
        self.lbIterations.setMinimumSize(QtCore.QSize(80, 30))
        self.lbIterations.setMaximumSize(QtCore.QSize(80, 30))
        self.lbIterations.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lbIterations.setTextFormat(QtCore.Qt.AutoText)
        self.lbIterations.setObjectName("lbIterations")
        self.gridLayout.addWidget(self.lbIterations, 0, 2, 1, 1)
        self.lbActivationFunction = QtWidgets.QLabel(self.layoutWidget3)
        self.lbActivationFunction.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbActivationFunction.sizePolicy().hasHeightForWidth())
        self.lbActivationFunction.setSizePolicy(sizePolicy)
        self.lbActivationFunction.setMinimumSize(QtCore.QSize(80, 30))
        self.lbActivationFunction.setMaximumSize(QtCore.QSize(80, 30))
        self.lbActivationFunction.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lbActivationFunction.setObjectName("lbActivationFunction")
        self.gridLayout.addWidget(self.lbActivationFunction, 1, 0, 1, 1)
        self.cbActivationFunction = QtWidgets.QComboBox(self.layoutWidget3)
        self.cbActivationFunction.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbActivationFunction.sizePolicy().hasHeightForWidth())
        self.cbActivationFunction.setSizePolicy(sizePolicy)
        self.cbActivationFunction.setMinimumSize(QtCore.QSize(94, 30))
        self.cbActivationFunction.setMaximumSize(QtCore.QSize(94, 30))
        self.cbActivationFunction.setStatusTip("")
        self.cbActivationFunction.setObjectName("cbActivationFunction")
        self.cbActivationFunction.addItem("")
        self.cbActivationFunction.addItem("")
        self.cbActivationFunction.addItem("")
        self.gridLayout.addWidget(self.cbActivationFunction, 1, 1, 1, 1)
        self.lbNeurons = QtWidgets.QLabel(self.layoutWidget3)
        self.lbNeurons.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbNeurons.sizePolicy().hasHeightForWidth())
        self.lbNeurons.setSizePolicy(sizePolicy)
        self.lbNeurons.setMinimumSize(QtCore.QSize(80, 30))
        self.lbNeurons.setMaximumSize(QtCore.QSize(80, 30))
        self.lbNeurons.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lbNeurons.setObjectName("lbNeurons")
        self.gridLayout.addWidget(self.lbNeurons, 1, 2, 1, 1)
        self.editNeurons = QtWidgets.QSpinBox(self.layoutWidget3)
        self.editNeurons.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editNeurons.sizePolicy().hasHeightForWidth())
        self.editNeurons.setSizePolicy(sizePolicy)
        self.editNeurons.setMaximumSize(QtCore.QSize(80, 30))
        self.editNeurons.setMinimum(1)
        self.editNeurons.setMaximum(999999999)
        self.editNeurons.setProperty("value", 9)
        self.editNeurons.setObjectName("editNeurons")
        self.gridLayout.addWidget(self.editNeurons, 1, 3, 1, 1)
        self.lbOptimizer = QtWidgets.QLabel(self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbOptimizer.sizePolicy().hasHeightForWidth())
        self.lbOptimizer.setSizePolicy(sizePolicy)
        self.lbOptimizer.setMinimumSize(QtCore.QSize(80, 30))
        self.lbOptimizer.setMaximumSize(QtCore.QSize(80, 30))
        self.lbOptimizer.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lbOptimizer.setObjectName("lbOptimizer")
        self.gridLayout.addWidget(self.lbOptimizer, 2, 0, 1, 1)
        self.cbOptimizer = QtWidgets.QComboBox(self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbOptimizer.sizePolicy().hasHeightForWidth())
        self.cbOptimizer.setSizePolicy(sizePolicy)
        self.cbOptimizer.setMinimumSize(QtCore.QSize(94, 30))
        self.cbOptimizer.setMaximumSize(QtCore.QSize(94, 30))
        self.cbOptimizer.setStatusTip("")
        self.cbOptimizer.setObjectName("cbOptimizer")
        self.cbOptimizer.addItem("")
        self.cbOptimizer.addItem("")
        self.cbOptimizer.addItem("")
        self.gridLayout.addWidget(self.cbOptimizer, 2, 1, 1, 1)
        self.lbAccuracy = QtWidgets.QLabel(self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbAccuracy.sizePolicy().hasHeightForWidth())
        self.lbAccuracy.setSizePolicy(sizePolicy)
        self.lbAccuracy.setMinimumSize(QtCore.QSize(80, 30))
        self.lbAccuracy.setMaximumSize(QtCore.QSize(80, 30))
        self.lbAccuracy.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lbAccuracy.setTextFormat(QtCore.Qt.AutoText)
        self.lbAccuracy.setObjectName("lbAccuracy")
        self.gridLayout.addWidget(self.lbAccuracy, 2, 2, 1, 1)
        self.editAccuracy = QtWidgets.QLineEdit(self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editAccuracy.sizePolicy().hasHeightForWidth())
        self.editAccuracy.setSizePolicy(sizePolicy)
        self.editAccuracy.setMinimumSize(QtCore.QSize(80, 30))
        self.editAccuracy.setMaximumSize(QtCore.QSize(80, 30))
        self.editAccuracy.setStatusTip("")
        self.editAccuracy.setClearButtonEnabled(True)
        self.editAccuracy.setObjectName("editAccuracy")
        self.gridLayout.addWidget(self.editAccuracy, 2, 3, 1, 1)
        self.editIterations = QtWidgets.QSpinBox(self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editIterations.sizePolicy().hasHeightForWidth())
        self.editIterations.setSizePolicy(sizePolicy)
        self.editIterations.setMaximumSize(QtCore.QSize(80, 30))
        self.editIterations.setMinimum(1)
        self.editIterations.setMaximum(999999999)
        self.editIterations.setProperty("value", 2500)
        self.editIterations.setObjectName("editIterations")
        self.gridLayout.addWidget(self.editIterations, 0, 3, 1, 1)
        self.btnSolve = QtWidgets.QPushButton(self.centralWidget)
        self.btnSolve.setEnabled(False)
        self.btnSolve.setGeometry(QtCore.QRect(370, 370, 120, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnSolve.sizePolicy().hasHeightForWidth())
        self.btnSolve.setSizePolicy(sizePolicy)
        self.btnSolve.setMinimumSize(QtCore.QSize(120, 30))
        self.btnSolve.setMaximumSize(QtCore.QSize(120, 30))
        self.btnSolve.setObjectName("btnSolve")
        self.frame = QtWidgets.QFrame(self.centralWidget)
        self.frame.setGeometry(QtCore.QRect(10, 290, 140, 110))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(140)
        sizePolicy.setVerticalStretch(110)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(140, 110))
        self.frame.setMaximumSize(QtCore.QSize(140, 110))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.layoutWidget4 = QtWidgets.QWidget(self.frame)
        self.layoutWidget4.setGeometry(QtCore.QRect(10, 0, 122, 112))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget4)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.cbTensorBoard = QtWidgets.QCheckBox(self.layoutWidget4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbTensorBoard.sizePolicy().hasHeightForWidth())
        self.cbTensorBoard.setSizePolicy(sizePolicy)
        self.cbTensorBoard.setMinimumSize(QtCore.QSize(120, 30))
        self.cbTensorBoard.setMaximumSize(QtCore.QSize(120, 30))
        self.cbTensorBoard.setStatusTip("")
        self.cbTensorBoard.setChecked(False)
        self.cbTensorBoard.setTristate(False)
        self.cbTensorBoard.setObjectName("cbTensorBoard")
        self.verticalLayout.addWidget(self.cbTensorBoard)
        self.cbPlotLoss = QtWidgets.QCheckBox(self.layoutWidget4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(120)
        sizePolicy.setVerticalStretch(30)
        sizePolicy.setHeightForWidth(self.cbPlotLoss.sizePolicy().hasHeightForWidth())
        self.cbPlotLoss.setSizePolicy(sizePolicy)
        self.cbPlotLoss.setMinimumSize(QtCore.QSize(120, 30))
        self.cbPlotLoss.setMaximumSize(QtCore.QSize(120, 30))
        self.cbPlotLoss.setStatusTip("")
        self.cbPlotLoss.setChecked(True)
        self.cbPlotLoss.setObjectName("cbPlotLoss")
        self.verticalLayout.addWidget(self.cbPlotLoss)
        self.cbPlotSolution = QtWidgets.QCheckBox(self.layoutWidget4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbPlotSolution.sizePolicy().hasHeightForWidth())
        self.cbPlotSolution.setSizePolicy(sizePolicy)
        self.cbPlotSolution.setMinimumSize(QtCore.QSize(120, 30))
        self.cbPlotSolution.setMaximumSize(QtCore.QSize(120, 30))
        self.cbPlotSolution.setStatusTip("")
        self.cbPlotSolution.setChecked(True)
        self.cbPlotSolution.setTristate(False)
        self.cbPlotSolution.setObjectName("cbPlotSolution")
        self.verticalLayout.addWidget(self.cbPlotSolution)
        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setSizeGripEnabled(False)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionProgram = QtWidgets.QAction(MainWindow)
        self.actionProgram.setObjectName("actionProgram")

        self.retranslateUi(MainWindow)
        self.cbModel.setCurrentIndex(0)
        self.cbActivationFunction.setCurrentIndex(0)
        self.cbOptimizer.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ODENet"))
        self.lbEquation.setText(_translate("MainWindow", "dy/dx ="))
        self.editEquation.setToolTip(_translate("MainWindow", "f(x, t)"))
        self.editEquation.setPlaceholderText(_translate("MainWindow", "right side of the first order ode"))
        self.lbInitialCondition.setText(_translate("MainWindow", "x0, y0 ="))
        self.editInitialCondition.setToolTip(_translate("MainWindow", "initial condition"))
        self.editInitialCondition.setPlaceholderText(_translate("MainWindow", "x0, y0"))
        self.lbSolutionInterval.setText(_translate("MainWindow", "Points"))
        self.cbPoints.setToolTip(_translate("MainWindow", "from where to load interval solution"))
        self.cbPoints.setItemText(0, _translate("MainWindow", "file"))
        self.cbPoints.setItemText(1, _translate("MainWindow", "direct"))
        self.editSolutionInterval.setPlaceholderText(_translate("MainWindow", "[x0, x1, x2,..., xN]"))
        self.btnLoadFromFile.setToolTip(_translate("MainWindow", "load interval from file"))
        self.btnLoadFromFile.setText(_translate("MainWindow", "..."))
        self.lbModel.setText(_translate("MainWindow", "Hola"))
        self.cbModel.setToolTip(_translate("MainWindow", "which optimization method to use"))
        self.cbModel.setItemText(0, _translate("MainWindow", "ChNN"))
        self.cbModel.setItemText(1, _translate("MainWindow", "LeNN"))
        self.cbModel.setItemText(2, _translate("MainWindow", "MLP"))
        self.cbModel.setItemText(3, _translate("MainWindow", "RBF"))
        self.lbIterations.setText(_translate("MainWindow", "Epochs:"))
        self.lbActivationFunction.setText(_translate("MainWindow", "Activation:"))
        self.cbActivationFunction.setToolTip(_translate("MainWindow", "which activation function to use"))
        self.cbActivationFunction.setItemText(0, _translate("MainWindow", "sigmoid"))
        self.cbActivationFunction.setItemText(1, _translate("MainWindow", "tanh"))
        self.cbActivationFunction.setItemText(2, _translate("MainWindow", "relu"))
        self.lbNeurons.setText(_translate("MainWindow", "Neurons: "))
        self.editNeurons.setToolTip(_translate("MainWindow", "number of neurons in the hidden layer (can\'t be greater than points in interval)"))
        self.lbOptimizer.setText(_translate("MainWindow", "Optimizer:"))
        self.cbOptimizer.setToolTip(_translate("MainWindow", "which optimization method to use"))
        self.cbOptimizer.setItemText(0, _translate("MainWindow", "rmsprop"))
        self.cbOptimizer.setItemText(1, _translate("MainWindow", "adam"))
        self.cbOptimizer.setItemText(2, _translate("MainWindow", "sgd"))
        self.lbAccuracy.setText(_translate("MainWindow", "Accuracy:"))
        self.editAccuracy.setToolTip(_translate("MainWindow", "desired solution precision"))
        self.editAccuracy.setText(_translate("MainWindow", "1e-2"))
        self.editAccuracy.setPlaceholderText(_translate("MainWindow", "1e-2"))
        self.editIterations.setToolTip(_translate("MainWindow", "number of neurons in the hidden layer (can\'t be greater than points in interval)"))
        self.btnSolve.setText(_translate("MainWindow", "Solve"))
        self.cbTensorBoard.setToolTip(_translate("MainWindow", "whatever to show a graphical solution or not"))
        self.cbTensorBoard.setText(_translate("MainWindow", "TensorBoard"))
        self.cbPlotLoss.setToolTip(_translate("MainWindow", "whatever to show the loss vs. epoch history or not"))
        self.cbPlotLoss.setText(_translate("MainWindow", "Plot Loss"))
        self.cbPlotSolution.setToolTip(_translate("MainWindow", "whatever to show a graphical solution or not"))
        self.cbPlotSolution.setText(_translate("MainWindow", "Plot Solution"))
        self.actionAbout.setText(_translate("MainWindow", "&About"))
        self.actionProgram.setText(_translate("MainWindow", "&Program"))

import ui.resources_rc