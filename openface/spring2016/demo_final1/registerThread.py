from PyQt4.QtCore import QThread
import register
import numpy as np
from PyQt4 import QtGui, QtCore, uic
import cv2

class Thread(QThread):
    send_img_signal = QtCore.pyqtSignal(np.ndarray)
    def __init__(self):
        super(Thread, self).__init__()
        self.stopFlag = False
        self.startSendFlag = True
        #self.registerDialog = register.RegisterDialog()

    def run(self):
        cap = cv2.VideoCapture(0)
 
        success,frame = cap.read()

        while success and (not self.stopFlag):
            success, frame = cap.read() 
            if success:
            	if self.startSendFlag:
                    self.send_img_signal.emit(frame)
                    self.startSendFlag = True              
                k = cv2.waitKey(1)

        if self.stopFlag:
        	cap.release()
        	print "release"


    def stopped(self):
    	self.stopFlag = True

    def setStartFlag(self):
    	self.startSendFlag=True

