#!/usr/bin/env python2

import argparse
import cv2
import itertools
import os
import pickle
import os
import time

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import dlib
detector = dlib.get_frontal_face_detector()

import sys
openface_root="/home/liuke/master/openface-master/"
#sys.path.insert(0,openface_root+'openface')
sys.path.insert(0,openface_root)

import openface
import openface.helper
from openface.data import iterImgs
from openface.alignment import NaiveDlib  # Depends on dlib.

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

modelDir = os.path.join(openface_root, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(img):
    
    #bb=detector(img,1)
    #if len(bb) > 0:
    #    bb= max(bb, key=lambda rect: rect.width() * rect.height())
    bb = align.getLargestFaceBoundingBox(img)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))

    alignedFace = align.alignImg("affine", args.imgDim, img, bb)
    if alignedFace is None:
        print("Unable to align image")

    rep = net.forwardImage(alignedFace)
    return bb,rep

def draw_dlib_rects(frame,rect,person,color):
	#for i,d in enumerate(rects):
	#	cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),color,2)
	#	cv2.putText(frame, 'DR:x%d' %i , (d.left(),d.top()-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness = 2, lineType=cv2.LINE_AA)
    cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(),rect.bottom()),color,2)
    cv2.putText(frame, person , (rect.left(),rect.top()-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness = 2, lineType=cv2.LINE_AA)


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

def infer(args):
    #img=cv2.imread('/home/liuke/graduationProject/openface-master/images/examples/carell.jpg')
    #img=cv2.imread('/home/liuke/Pictures/girls.jpg')
    # img=cv2.imread('/home/liuke/Pictures/test1.jpg')
    img=cv2.imread("/home/liuke/master/openface-master/Gene_SimMatrix/data/Rachel/Rachel_197.jpg")
    with open(args.classifierModel, 'r') as f:
        (le,svm)=pickle.load(f)
    
    vis=img.copy() 
    t=clock()

    bb,rep = getRep(img)
        
    predictions = svm.predict_proba(rep)[0]
    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = predictions[maxI]
    draw_dlib_rects(vis,bb,person,(0,255,0))
        
    dt=clock()-t
    draw_str(vis,(20,20),'time: %.1f ms' % (dt*1000))
    print("Predict {} with {:.2f} confidence.".format(person, confidence))
    print("bb.left,bb.right,bb.top,bb.bottom: ",bb.left(),bb.right(),bb.top(),bb.bottom())

    cv2.imshow('facedr',vis)
    cv2.waitKey(100)
    time.sleep(50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dlibFaceMean', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "mean.csv"))
    parser.add_argument('--dlibFacePredictor', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir,
                                             "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str,
                        help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--classifierModel', default=os.path.join(openfaceModelDir, "celeb-classifier.nn4.v1.pkl"))
    args = parser.parse_args()



    align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
    net = openface.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)
        
    infer(args)
