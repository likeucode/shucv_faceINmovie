#!/usr/bin/env python2
#
# Using openface for face image retrieval .
# liuke
# 2016/03/04
#

import time

start = time.time()
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import cv2
import itertools
import os
import shutil
import random
import scipy as sp
from scipy import io as spio
import numpy as np
np.set_printoptions(precision=2)
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "../"))

import openface

def preprocessDB(img_root,other_path):
    dirout=os.listdir(img_root)
    dirout.sort()
    for subdir in dirout:
        subdir_path=os.path.join(img_root,subdir)
        subdirout=os.listdir(subdir_path)
        for img in subdirout:
            img_loc=os.path.join(subdir_path,img)
            bgrImg = cv2.imread(img_loc)
            if bgrImg is None:
                print("Unable to load image: {}".format(img_loc))
                shutil.move(img_loc,other_path) 
                continue     
            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

            bb = align.getLargestFaceBoundingBox(rgbImg)
            if bb is None:
                print("Unable to find a face: {}".format(img_loc))
                shutil.move(img_loc,other_path)
                continue        
 
            alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                print("Unable to align image: {}".format(img_loc))
                shutil.move(img_loc,other_path)      

def getRep(imgPath):
    
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
         
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
  
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
   
    rep = net.forward(alignedFace)
    return rep

if __name__=='__main__':
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

    if args.verbose:
        print("Argument parsing and loading libraries took {} seconds.".format(
            time.time() - start))

    start = time.time()
    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
   
    
    dataset_path='/home/byx/Dataset/BothDB/'
    
    dirout1=os.listdir(dataset_path)
    dirout1.sort()
    print("number  of initial subdirs: ",len(dirout1))
    num=0
    fea_mat=np.zeros([1,128])
    fea_mat=np.reshape(fea_mat,(1,128))
    id_vec=np.zeros(1)
    names=np.array('head')
    for actor in dirout1:
        num=num+1       
        subdir_path=os.path.join(dataset_path,actor)
        dirout2=os.listdir(subdir_path)
        dirout2.sort()
        print("images of this actor is {}.".format(len(dirout2)))
        for img in dirout2:
            actor_img=os.path.join(subdir_path,img)
            print("Processing image {}.".format(actor_img))
            feature=getRep(actor_img)
            feature=np.reshape(feature,(1,128))
            fea_mat=np.append(fea_mat,feature,axis=0)
            # id_vec=np.append(id_vec,num)
            names=np.append(names,actor)
            print("Shape of fea_mat is {}.".format(fea_mat.shape))
            print("ID of image is {}.".format(id_vec[num]))
            print("Name of actor is {}.".format(actor))

    id_vec=id_vec[1:]
    names=names[1:]
    fea_mat=fea_mat[1:,:]
    spio.savemat("openface_fea.mat",{'feature':fea_mat,'id':id_vec,'names':names})

    Xtrain,Xtest,Ytrain,Ytest=cross_validation.train_test_split(fea_mat,id_vec,test_size=0.2,random_state=0)
    knn=KNeighborsClassifier(n_neighbors=10)
    # KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
    #  p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)

    knn.fit(Xtrain,Ytrain)
    knn.score(Xtest,Ytest)
    
    ###Parameter estimation using grid search with CV###
    param_grid=[{'n_neighbors':[5,10,20,50],'weights':['uniform','distance'],
    'algorithm':['auto','brute'],'p':[1,2]},
    {'n_neighbors':[5,10,20,50],'weights':['uniform','distance'],
    'algorithm':['ball_tree','kd_tree'],'leaf_size':[20,30,40],'p':[1,2]}]

    clf=GridSearchCV(KNeighborsClassifier(),param_grid,cv=3)
    clf.fit(Xtrain,Ytrain)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Ytest, clf.predict(Xtest)
    print(classification_report(y_true, y_pred))
    print()
            
           
