import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import idx2numpy

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
    # Converting into an array: 28 entries of size 28 (each entry is a row) #
    arrayImage=np.asarray(resizedImage)
    # Normalizing every value (instead of integers, real-valued inputs in [0;1]) #
    arrayImage=tf.keras.utils.normalize(arrayImage, axis=1)
    arrayImage=np.expand_dims(arrayImage,axis=0)
    formattedImage=np.vstack([arrayImage])
    #formattedImage=[]
    #formattedImage.append(0x00000803)
    #formattedImage.append(1)
    #formattedImage.append(28)
    #formattedImage.append(28)
    #for i in range(28):
        #for j in range(28):
            #formattedImage.append(arrayImage[i][j])
    #formattedImage=np.array(formattedImage)
    #formattedImage=np.vstack(formattedImage)
    return formattedImage

def search_maxindex(list):
    maxindex=0
    maxvalue=list[0]
    for k in range(len(list)):
        if list[k]>maxvalue:
            maxindex=k
            maxvalue=list[k]
    return maxindex

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
    def __init__(self,userModel):
        super(Qtw.QWidget,self).__init__()

        self.nnLayout=Qtw.QGridLayout(self)
        self.setLayout(self.nnLayout)

        # Current model variable for NN, chosen by user # 
        self.currentModelNNWid=userModel
        self.epochNumber=5

        # Textbox and its label for number of epochs input # 
        self.epochsLabel=Qtw.QLabel('Desired # of epochs:')
        self.nnLayout.addWidget(self.epochsLabel,0,2)
        self.textboxEpochs=Qtw.QLineEdit(self)
        self.nnLayout.addWidget(self.textboxEpochs, 1, 2)

        # Showing "Train NN" push button #
        self.trainButton=Qtw.QPushButton('Train NN', self)
        self.nnLayout.addWidget(self.trainButton, 2, 2)

        # Connecting button to click, textbox to enter key # 
        self.trainButton.clicked.connect(self.on_click_train)
        self.readEpochsShortcut=Qtw.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self.textboxEpochs)
        self.readEpochsShortcut.activated.connect(self.on_click_updateepochs)

    def on_click_train(self):
        """ Trains NN used to recognize handwritten numbers
        number of epochs can be modified, could be fun to 
        be able to choose nb of hidden layers too, and other parameters """
        train_NN(self.currentModelNNWid,self.epochNumber)

    def on_click_updateepochs(self):
        """ Updates the desired # of epochs for in-app training"""
        self.epochNumber=int(self.textboxEpochs.text())

class ImportZone(Qtw.QWidget):
    """ Widget to import pre-trained and pre-existing models """
    def __init__(self):
        super(Qtw.QWidget,self).__init__()

        self.izLayout=Qtw.QGridLayout(self)
        self.setLayout(self.izLayout)

        # Text label #
        self.importLabel=Qtw.QLabel('Select model from file:')
        self.izLayout.addWidget(self.importLabel,0,0)

        # Text box #
        self.fileSpace=Qtw.QLineEdit()
        self.izLayout.addWidget(self.fileSpace,1,0)

        # Browse button #
        self.browseButton=Qtw.QPushButton('Browse',self)
        self.izLayout.addWidget(self.browseButton,2,0)

        # Connecting button to click #
        self.browseButton.clicked.connect(self.on_click_browse)

    def openFileNamesDialog(self):
        # Modify to only allow .model format 
        options = Qtw.QFileDialog.Options()
        options |= Qtw.QFileDialog.DontUseNativeDialog
        files, _ = Qtw.QFileDialog.getOpenFileNames(self,"Browse", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            print(files)
    
    def on_click_browse(self):
        self.openFileNamesDialog()

class ReadZone(Qtw.QWidget):
    """ Widget to actually draw and read numbers"""
    def __init__(self):

        super(Qtw.QWidget,self).__init__()
        self.readLayout=Qtw.QGridLayout()
        self.setLayout(self.readLayout)

        # Showing instuctions label #
        self.instructLabel=Qtw.QLabel("Draw a number between 0 and 9 in the box below. \nTry to center it as much as possible.")
        self.readLayout.addWidget(self.instructLabel,0,0)

        # Showing paint zone # 
        self.pZone=PaintZone()
        self.readLayout.addWidget(self.pZone,1,0)

        # Showing "Read" push button #
        self.readButton=Qtw.QPushButton('Read', self)
        self.readLayout.addWidget(self.readButton,2,0)

        # Showing "Reset" push button # 
        self.resetButton=Qtw.QPushButton('Reset', self)
        self.readLayout.addWidget(self.resetButton, 3, 0)

        # Showing result label #
        self.rzoneLabel=Qtw.QLabel('Prediction is:')
        self.readLayout.addWidget(self.rzoneLabel,4,0)

        # Showing result zone #
        self.rZone=Qtw.QLabel('')
        self.readLayout.addWidget(self.rZone,5,0)

        # Connecting buttons to click #
        self.readButton.clicked.connect(self.on_click_read)
        self.resetButton.clicked.connect(self.on_click_reset)
       

    def on_click_read(self):
        """ Feeds drawn character to neural network trained on
        MNIST database"""
        try:
            self.pZone.saveImage("image.jpeg", "JPEG")
            toPredict=process_img('image.jpeg')
            self.currentModel=self.nnZone.currentModelNNWid
            result=self.nnZone.currentModelNNWid.predict(toPredict)
            predictedNb=search_maxindex(result[0])
            self.rZone.setText(str(predictedNb))
        except: self.rZone.setText('NN not trained')


    def on_click_reset(self):
        self.rZone.setText('')
        self.pZone.clearImage()
        try:
            os.remove('image.jpeg')
        except: None


class MainWindow(Qtw.QWidget):
    """ Main (and sole) window of the app """
    def __init__(self):
        super(Qtw.QWidget,self).__init__()

        # Graphic initialization #
        self.setFixedSize(800,700)
        self.setWindowTitle("Handwritten numbers reader")
        self.mainLayout=Qtw.QGridLayout(self)
        self.setLayout(self.mainLayout)

        #Showing read zone # 
        self.rZone=ReadZone()
        self.mainLayout.addWidget(self.rZone,0,0)

        # Showing NN zone 
        self.nnZone=NNZone(tf.keras.models.Sequential())
        self.mainLayout.addWidget(self.nnZone,0,1)
        self.currentModel=self.nnZone.currentModelNNWid

        # Showing import zone #
        self.iZone=ImportZone()
        self.mainLayout.addWidget(self.iZone,0,2)



if __name__ == '__main__':
    app=Qtw.QApplication(sys.argv)
    mainWin=MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


