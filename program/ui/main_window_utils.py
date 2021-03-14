
author = 'Pedro Pablo'


from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QProgressBar, QLabel


def message(widget, msg):
    widget.showMessage(msg, -1)

def update_status_bar(status, progress, label):
    progress_bar, lb_loss = None, None
    
    if progress is None:
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        progress_bar.setValue(0)
        progress_bar.width()
        progress_bar.setVisible(True)
        lb_loss = QLabel('   loss: 0.0')
        lb_loss.setVisible(True)
        rgSpacer = QLabel('   ')
        rgSpacer.setVisible(True)
        status.addWidget(lb_loss, 1)
        status.addWidget(progress_bar, 4)
        status.addWidget(rgSpacer, 1)
        
        hide([progress_bar])
    else:
        label.setText('   loss : 0.0')
        progress.setValue(0)
        progress.setVisible(True)
        label.setVisible(True)

    return progress_bar, lb_loss

def hide(widgets):
    for widget in widgets:
        widget.setVisible(False)

def show(widgets):
    for widget in widgets:
        widget.setVisible(True)

def integer_validator():
    # Regular Expression for `int` representations
    re = QRegExp("[1-9]+[0-9]+")
    return QRegExpValidator(re)

def float_validator():
    # Regular Expression for `float` representations
    re = QRegExp(
        "^[+]?([0-9]+(\.[0-9]+)?|\.[0-9]+)([eE][-+]?[0-9]+)?$")
    return QRegExpValidator(re)
    

def initial_condition_validator():
    # Regular Expression for `initial condition`
    re = QRegExp(
        "^[+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+)([eE][-+]?[0-9]+)?[]?[,][]? [+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+)([eE][-+]?[0-9]+)?$")
    return QRegExpValidator(re)

def enable_widgets(widgets, value):
    for widget in widgets:
        widget.setEnabled(value)
    if widgets[0].currentIndex() == 0 and value == True:
        widgets[1].setEnabled(False)
