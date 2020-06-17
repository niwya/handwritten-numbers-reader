import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# App builder # 
from PyQt5 import QtWidgets as Qtw
from PyQt5 import QtCore, QtGui

# Path of image to process - adapt to your use #
path = r'C:\Users\Chloe\Documents\GitRepositories\Handwritten_num_reader\image.jpeg'

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

## Processing images in order to feed them to NN ##

def process_img(imagePath):
    """ Must return a 28x28 px image on the form of a list, ready to feed to NN (scale ratio : /50) """
    resizedImage=cv2.resize(cv2.imread(imagePath,0), (28,28), interpolation=cv2.INTER_NEAREST)
    # Applying negative filter #
    resizedImage=cv2.bitwise_not(resizedImage) 
    # Convert into an array: 28 entries of size 28 (each entry is a row) #
    arrayImage=np.asarray(resizedImage)
    formattedImage=[]
    formattedImage.append(0x00000803)
    formattedImage.append(1)
    formattedImage.append(28)
    formattedImage.append(28)
    for i in range(28):
        for j in range(28):
            formattedImage.append(arrayImage[i][j])
    return formattedImage

## Application layout ##

class PaintZone(Qtw.QWidget):
    """ Paint zone widget, 140x140 px high and wide, allowing the user to draw
    (must be a number between 0 and 9) 
    Based on stackoverflow.com/questions/48046462/pyqt5-i-save-the-image-but-it-is-always-empty """
    def __init__(self):
        super(Qtw.QWidget,self).__init__()
        h = 140
        w = 140
        self.myPenWidth = 8
        self.myPenColor = QtCore.Qt.black
        self.image = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
        self.path = QtGui.QPainterPath()
        self.clearImage()

    def clearImage(self):
        self.path = QtGui.QPainterPath()
        self.image.fill(QtCore.Qt.white)
        self.update()

    def saveImage(self, fileName, fileFormat):
        self.image.save(fileName, fileFormat)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())

    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())

    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        p = QtGui.QPainter(self.image)
        p.setPen(QtGui.QPen(self.myPenColor,
                      self.myPenWidth, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                      QtCore.Qt.RoundJoin))
        p.drawPath(self.path)
        p.end()
        self.update()

class NNZone(Qtw.QWidget):
    """ Widget used to input parameters and train the NN """
    def __init__(self):
        super(Qtw.QWidget,self).__init__()

        self.nnLayout=Qtw.QGridLayout(self)
        self.setLayout(self.nnLayout)

        # Current model variable for NN # 
        self.currentModelNNWid=None
        self.epochNumber=5

        # Showing "Train NN" push button #
        self.trainButton=Qtw.QPushButton('Train NN', self)
        self.nnLayout.addWidget(self.trainButton, 2, 2)

        # Connecting button to click # 
        self.trainButton.clicked.connect(self.on_click_train)

    def on_click_train(self):
        """ Trains NN used to recognize handwritten numbers
        number of epochs can be modified, could be fun to 
        be able to choose nb of hidden layers too, and other parameters """
        train_NN(self.currentModelNNWid,self.epochNumber)

class MainWindow(Qtw.QWidget):
    """ Main (and sole) window of the app """
    def __init__(self):
        super(Qtw.QWidget,self).__init__()

        # Graphic initialization #
        self.setFixedSize(500,500)
        self.setWindowTitle("Handwritten numbers reader")
        self.mainLayout=Qtw.QGridLayout(self)
        self.setLayout(self.mainLayout)

        # Current model variable for NN #
        self.currentModel=None

        # Showing instuctions label #
        self.instructLabel=Qtw.QLabel("Draw a number between 0 and 9 in the box below. \nTry to center it as much as possible.")
        self.mainLayout.addWidget(self.instructLabel,0,0)

        # Showing paint zone # 
        self.pZone=PaintZone()
        self.mainLayout.addWidget(self.pZone,1,0)

        # Showing result zone #
        self.rZone=Qtw.QLabel('')
        self.mainLayout.addWidget(self.rZone,1,1)

        # Showing NN zone #
        self.nnZone=NNZone()
        self.mainLayout.addWidget(self.nnZone,2,1)

        # Showing "Read" push button #
        self.readButton=Qtw.QPushButton('Read', self)
        self.mainLayout.addWidget(self.readButton,2,0)

        # Showing "Reset" push button # 
        self.resetButton=Qtw.QPushButton('Reset', self)
        self.mainLayout.addWidget(self.resetButton, 0, 1)

        # Connecting buttons to click/enter key #
        self.readButton.clicked.connect(self.on_click_read)
        self.resetButton.clicked.connect(self.on_click_reset)
        self.readButtonShortcut=Qtw.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self.readButton)
        self.readButtonShortcut.activated.connect(self.on_click_read)

    def on_click_read(self):
        """ Feeds drawn character to neural network trained on
        MNIST database"""
        self.pZone.saveImage("image.jpeg", "JPEG")
        toPredict=process_img('image.jpeg')
        #self.currentModel=self.nnZone.currentModelNNWid
        #result=self.nnZone.currentModelNNWid.predict(toPredict)
        #self.rZone.setText(result)


    def on_click_reset(self):
        self.rZone.setText('')
        self.pZone.clearImage()
        try:
            os.remove('image.jpeg')
        except: None


if __name__ == '__main__':
    app=Qtw.QApplication(sys.argv)
    mainWin=MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


