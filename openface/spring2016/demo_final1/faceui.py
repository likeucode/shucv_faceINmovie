# -*- coding:utf-8 -*-  
import sys
import time
from PyQt4 import QtGui, QtCore, uic

from sklearn.neighbors import KNeighborsClassifier 
import argparse
import cv2
import itertools
import os
import shutil
import random
import scipy as sp
from scipy import io as spio
import numpy as np
# np.set_printoptions(precision=5)
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "../"))

import openface

import register

import registerThread

class MyDialog(QtGui.QDialog):
    def __init__(self):
        super(MyDialog, self).__init__()
        self.registerDialog = register.RegisterDialog()
        
        uic.loadUi("deepface.ui", self)
        self.reg_thread = registerThread.Thread()
        self.create_signal_slot()
 
        self.iscameraWorking = False
        self.isfileWorking = False
        self.ishasFile = False
        # self.isReg=False

        self.cnt=0

        if os.path.exists('./register_person.mat'):
            reg_data=spio.loadmat("register_person.mat")
            self.reg_X=reg_data['feature']
            self.reg_id=reg_data['id'].astype(int)
            print("reg_id:{}".format(self.reg_id))
            
            self.reg_names=list(reg_data['names'])
            

        else:
            self.reg_X=np.zeros([1,128])
            self.reg_id=np.zeros(1)
            self.reg_names=np.array('head')
            
        #r = self.exec_()


    def create_signal_slot(self):
        self.connect(self.fileButton, QtCore.SIGNAL('clicked()'), self.onfileButton)
        self.connect(self.cameraButton, QtCore.SIGNAL('clicked()'), self.oncameraButton)
        self.connect(self.registerButton, QtCore.SIGNAL('clicked()'), self.onregisterButton)
        self.connect(self.startButton, QtCore.SIGNAL('clicked()'), self.onstartButton)
 
        self.registerDialog.send_register_signal.connect(self.accept_register_slot)
        self.reg_thread.send_img_signal.connect(self.accept_img_slot)


    def onfileButton(self):
        self.file_name = QtGui.QFileDialog.getOpenFileName(self, "Select video" , "./")
        self.ishasFile = True
        print self.file_name

    def oncameraButton(self):

        print("***Train register knn classifier***")
        X=self.reg_X[1:,:]
        print("X:{}".format(X))
        Y= self.reg_id[0]
        Y=Y[1:]-1
        print("Y:{}".format(Y))
        names=self.reg_names[1:]
        # names=names.sort()
        reg_knn=KNeighborsClassifier(n_neighbors=8,weights='distance',p=2)
        reg_knn.fit(X,Y)

        cap = cv2.VideoCapture(0)
        if self.iscameraWorking == False:
            self.cameraButton.setText("Close")
            # cap = cv2.VideoCapture(0)
            self.iscameraWorking = True
            
            success,frame = cap.read()

            while success and self.iscameraWorking :
                start=time.time()
                success, frame = cap.read() 
                if success:
                    img=frame.copy()
                    bb,rep=getRep(img)
                    if bb is None:
                        print "Can't find any face in this picture"
                    else:
                        if rep is 0:
                            print "Get rep failed..."
                        else:
                            rep=np.reshape(rep,(1,128))
                            idx=reg_knn.predict(rep)
                            # print("label is {} ".format(idx))
                            proba=reg_knn.predict_proba(rep)
                            actor=names[idx]
                            self.namelineEdit.setText(actor)
                            self.timelineEdit.setText(str(round(time.time()-start,3)))
                            self.confidencelineEdit.setText(str(round(max(proba[0]),2)))
                            # print("Proba is {} ".format(proba))
                            
                            

                            draw_dlib_rects(frame,bb,actor,(0,255,0))
                    image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                    pixmap = QtGui.QPixmap.fromImage(image)
                    self.showlabel.setPixmap(pixmap)
                    k = cv2.waitKey(1)
        else:
            self.cameraButton.setText("Camera")
            self.iscameraWorking = False
            cap.release()
            self.showlabel.clear()

    def onregisterButton(self):
        print "register"
        self.cnt=0

        self.registerDialog.show()
        self.reg_thread.run()

    def accept_register_slot(self, register_message):
        print register_message['name']
        print register_message['gender']
        print register_message['age']

        actor=register_message['name']

        self.reg_names=np.append(self.reg_names,str(actor))
        spio.savemat("register_person.mat",{'feature':self.reg_X,'id':self.reg_id,'names':self.reg_names})
        print("Save mat file sucessfully!")

        self.reg_thread.stopped()
        self.reg_thread.quit()
        self.showlabel.clear()

    def accept_img_slot(self, img):
        frame=img.copy()
        bb=detectFace(img)
        if bb is None:
            print "Can't find any face in this picture"
        else:
            self.cnt=self.cnt+1
            if self.cnt<=5:
                bb1,rep=getRep(img)
                self.reg_X=np.append(self.reg_X,np.reshape(rep,(1,128)),axis=0)
                
            elif self.cnt==6:
                self.reg_id=np.append(self.reg_id,np.repeat(max(self.reg_id)+1,5))
            else:
                print("OK!")
                draw_str(frame,(0,0),"OK")

            draw_dlib_rects(frame,bb,"Unknow",(0,255,0))
            image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap = QtGui.QPixmap.fromImage (image)
            self.showlabel.setPixmap(pixmap)        
            # self.reg_thread.startSendFlag = True
            self.reg_thread.setStartFlag()

    def onstartButton(self):

        cap = cv2.VideoCapture(str(self.file_name))

        if self.isfileWorking == False and self.ishasFile == True:
            self.ishasFile = False
            self.startButton.setText("Close")

            # cap = cv2.VideoCapture(str(self.file_name))

            self.isfileWorking = True
            data=spio.loadmat("openface_fea.mat")
            X=data['feature']
            id=data['id'].astype(int)-1
            Y=id[0,:]
            name=list(set(data['names']))
            name.sort()
            print("***Train knn classifier***")
            knn=KNeighborsClassifier(n_neighbors=20,weights='distance',p=2)
            knn.fit(X,Y)

            success,frame = cap.read()

            while success and self.isfileWorking :
            	start=time.time()
                success, frame = cap.read() 
                
                if success:
                    img=frame.copy()
                   
                    bb,rep=getRep(img)
                    if bb is None:
                        print "Can't find any face in this picture"
                    else:
                        if rep is 0:
                            print "Get rep failed..."
                        else:
                            rep=np.reshape(rep,(1,128))
                            idx=knn.predict(rep)
                            # print("label is {} ".format(idx))
                            proba=knn.predict_proba(rep)
                            actor=name[idx]
                            self.namelineEdit.setText(actor)
                            self.timelineEdit.setText(str(round(time.time()-start,3)))
                            self.confidencelineEdit.setText(str(round(max(proba[0]),2)))
                            # print("Proba is {} ".format(proba))
                            
                            

                            draw_dlib_rects(frame,bb,actor,(0,255,0))
                    image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                    pixmap = QtGui.QPixmap.fromImage(image)
                    self.showlabel.setPixmap(pixmap)
                    k = cv2.waitKey(5)
        else:
            self.ishasFile = False
            self.startButton.setText("Start")
            self.isfileWorking = False
            cap.release()
            self.showlabel.clear()


def draw_dlib_rects(frame,rect,person,color):
    #for i,d in enumerate(rects):
    #	cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),color,2)
    #	cv2.putText(frame, 'DR:x%d' %i , (d.left(),d.top()-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness = 2, lineType=cv2.LINE_AA)
    cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(),rect.bottom()),color,1)
    cv2.putText(frame, person , (rect.left(),rect.top()-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness = 2, lineType=cv2.LINE_AA)


def draw_str(dst, (x, y), s):
    # cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x+10, y+10), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0), thickness = 2,lineType=cv2.LINE_AA)

def detectFace(bgrImg):
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        print("Unable to find a face")

    else:
        return bb

def getRep(bgrImg):
    rep=0
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        print("Unable to find a face")
        return bb,rep
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        print("Unable to align image")
        return bb,rep

    rep = net.forward(alignedFace)
    return bb,rep

if __name__ == '__main__':
    modelDir = os.path.join(fileDir, '..', 'models')
    dlibModelDir = os.path.join(modelDir, 'dlib')
    openfaceModelDir = os.path.join(modelDir, 'openface')

    parser = argparse.ArgumentParser()

    #parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

    app = QtGui.QApplication(sys.argv)
    demo = MyDialog()
    demo.show()
    app.exec_()
