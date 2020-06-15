import sys
import numpy as np
#import tensorflow as tf

# App builder # 
from PyQt5 import QtWidgets as Qtw
from PyQt5 import QtCore, QtGui


### Handwritten character reader ###

## Application layout ##

class PaintZone(Qtw.QWidget):
    """ Paint zone widget, 140x140 px high and wide, allowing the user to draw
    (must be a number between 0 and 9) 
    Based on www.learnpyqt.com/courses/custom-widgets/bitmap-graphics/ tutorial """
    def __init__(self):
        super().__init__()
        self.painterLayout=Qtw.QGridLayout(self)
        self.setLayout(self.painterLayout)

        self.painterWid = Qtw.QLabel()
        canvas = QtGui.QPixmap(140, 140)
        canvas.fill(QtGui.QColor('#ffffff')) #fill with white, otherwise appears black
        self.painterWid.setPixmap(canvas)

        self.last_x, self.last_y = None, None

        # Showing canvas #
        self.painterLayout.addWidget(self.painterWid)
        self.painterWid.setAlignment(QtCore.Qt.AlignCenter)

        # Showing "Reset" button #
        self.resetButton=Qtw.QPushButton('Reset', self)
        self.painterLayout.addWidget(self.resetButton, 1, 0)
        self.resetButton.clicked.connect(self.on_click_reset)

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.painterWid.pixmap())
        defaultPen=painter.pen()
        defaultPen.setWidth(6)
        painter.setPen(defaultPen)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def on_click_reset(self):
        print("Resetting done")

    


class MainWindow(Qtw.QWidget):
    """ Main (and sole) window of the app """
    def __init__(self):
        super(Qtw.QWidget,self).__init__()

        self.setWindowTitle("Handwritten numbers reader")
        
        self.mainLayout=Qtw.QGridLayout(self)
        self.setLayout(self.mainLayout)

        # Showing instuctions label #
        self.instructLabel=Qtw.QLabel("Draw a number between 0 and 9 in the box below. \nTry to center it as much as possible.")
        self.mainLayout.addWidget(self.instructLabel,0,0)

        # Showing paint zone # 
        self.pZone=PaintZone()
        self.mainLayout.addWidget(self.pZone,1,0)

        # Showing "Read" push button #
        self.readButton=Qtw.QPushButton('Read', self)
        self.mainLayout.addWidget(self.readButton,2,0)

        # Connecting button to click/enter key #
        self.readButton.clicked.connect(self.on_click_read)
        self.readButtonShortcut=Qtw.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self.readButton)
        self.readButtonShortcut.activated.connect(self.on_click_read)

        # Showing zone where to diplay result of neural network #
        self.resultZone=Qtw.QWidget()

    def on_click_read(self):
        """ on_click function feeds drawn character to neural network trained on
        MNIST database"""
        print("Reading done")
    

if __name__ == '__main__':
    app=Qtw.QApplication(sys.argv)
    mainWin=MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


