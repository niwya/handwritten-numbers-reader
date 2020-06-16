import sys
import numpy as np
import tensorflow as tf

# App builder # 
from PyQt5 import QtWidgets as Qtw
from PyQt5 import QtCore, QtGui


### Handwritten character reader ###

## Training NN ## 

def train_NN(model, epochs):
    mnist=tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=tf.keras.utils.normalize(x_train, axis=1)
    x_test=tf.keras.utils.normalize(x_test, axis=1)
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs)
    val_loss, val_acc=model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

## Processing images in order to feed them to NN


## Application layout ##

class PaintZone(Qtw.QWidget):
    """ Paint zone widget, 140x140 px high and wide, allowing the user to draw
    (must be a number between 0 and 9) 
    Based on www.learnpyqt.com/courses/custom-widgets/bitmap-graphics/ """
    def __init__(self):
        super().__init__()
        self.painterLayout=Qtw.QGridLayout(self)
        self.setLayout(self.painterLayout)

        self.painterWid = Qtw.QLabel()
        self.canvas = QtGui.QPixmap(140, 140) #
        self.canvas.fill(QtGui.QColor('#ffffff')) #Fill with white, otherwise paint zone appears black
        self.painterWid.setPixmap(self.canvas)

        self.last_x, self.last_y = None, None

        # Showing canvas #
        self.painterLayout.addWidget(self.painterWid)
        #self.painterWid.setAlignment(QtCore.Qt.AlignCenter) #Bug: centering widget does not center the "brush proc zone"

    def mouseMoveEvent(self, e):
        if self.last_x is None: #First event
            self.last_x = e.x()
            self.last_y = e.y()
            return #Ignore the first time

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

        # Showing result zone #
        self.rZone=Qtw.QLabel('')
        self.mainLayout.addWidget(self.rZone,1,1)

        # Showing "Read" push button #
        self.readButton=Qtw.QPushButton('Read', self)
        self.mainLayout.addWidget(self.readButton,2,0)

        # Showing "Train NN"
        self.trainButton=Qtw.QPushButton('Train NN', self)
        self.mainLayout.addWidget(self.trainButton, 2, 1)


        self.resetButton=Qtw.QPushButton('Reset', self)
        self.mainLayout.addWidget(self.resetButton, 0, 1)
        # Connecting button to click/enter key #
        self.readButton.clicked.connect(self.on_click_read)
        self.trainButton.clicked.connect(self.on_click_train)
        self.resetButton.clicked.connect(self.on_click_reset)
        self.readButtonShortcut=Qtw.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self.readButton)
        self.readButtonShortcut.activated.connect(self.on_click_read)

        # Showing zone where to diplay result of neural network #
        self.resultZone=Qtw.QWidget()

    def on_click_read(self):
        """ Feeds drawn character to neural network trained on
        MNIST database"""
        print("Reading done")
    
    def on_click_train(self):
        """ Trains NN used to recognize handwritten numbers
        number of epochs can be modified, could be fun to 
        be able to choose nb of hidden layers too, and other parameters """
        myModel=None
        nTrainings=5
        train_NN(myModel,nTrainings)

    def on_click_reset(self):
        self.pZone.canvas.fill(QtGui.QColor('#ffffff'))
        self.pZone.painterWid.setPixmap(self.pZone.canvas)   
        self.rZone.setText('')
    

if __name__ == '__main__':
    app=Qtw.QApplication(sys.argv)
    mainWin=MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


