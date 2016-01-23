#!/usr/bin/env python2
#
# Using the result of face tracking to build image database .
# liuke
# 2016/01/22
#

import time

start = time.time()

import argparse
import cv2
import itertools
import os
import shutil
import random

import numpy as np
np.set_printoptions(precision=2)
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

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
   
    
    img_path='/home/byx/openface2016/openface/kelly/tracking/Result/'
    other_path='/home/byx/openface2016/openface/kelly/tracking/other/'
    #preprocess the db
    '''
    print("Preprocessing the tracking result...")
    preprocessDB(img_path,other_path)
    print("OK,finish preprocessing the tracking result!")
    '''

    dirout1=os.listdir(img_path)
    dirout1.sort()
    dis_thr=0.8 
    print("number  of initial subdirs: ",len(dirout1))

    num=0
    while len(dirout1)>10:
        base_subdir=dirout1[num]
        base_subdir_path=os.path.join(img_path,base_subdir)
        base_dirout=os.listdir(base_subdir_path)
        base_dirout.sort()
        base_dirout=np.array(base_dirout)
        print("Current base subdir is {}.".format(base_subdir))
        for next_subdir in dirout1[num+1:] :
            # next_subdir=dirout1[num+1]
            next_subdir_path=os.path.join(img_path,next_subdir)
            next_dirout=os.listdir(next_subdir_path)
            next_dirout.sort()
            next_dirout=np.array(next_dirout)
            print("----Processing next subdir {}, Current base is {}".format(next_subdir,base_subdir))

            if len(base_dirout)>len(next_dirout):
                base_part_imgs_idx=random.sample(range(len(base_dirout)),len(next_dirout))
                base_part_imgs=base_dirout[base_part_imgs_idx]
                eat_norm=0.0
                for i in range(len(next_dirout)):
                    rep1 =getRep(os.path.join(base_subdir_path,base_part_imgs[i]))
                    rep2 =getRep(os.path.join(next_subdir_path,next_dirout[i]))
                    diff =rep1-rep2
                    dis =np.dot(diff,diff)
                    eat_norm +=dis

                if eat_norm/len(next_dirout)<dis_thr:
                    for i in range(len(next_dirout)):
                        shutil.move(os.path.join(next_subdir_path,next_dirout[i]),base_subdir_path)
                        print('************************************************')
                        print("++++++++Moving {} to {}.".format(next_dirout[i],base_subdir_path))
                        print('************************************************')
                    os.removedirs(next_subdir_path)
                else:
                    pass

            else:
                next_part_imgs_idx=random.sample(range(len(next_dirout)),len(base_dirout))
                next_part_imgs=next_dirout[next_part_imgs_idx]
                eat_norm=0.0
                for i in range(len(base_dirout)):
                    rep1 =getRep(os.path.join(next_subdir_path,next_part_imgs[i]))
                    rep2 =getRep(os.path.join(base_subdir_path,base_dirout[i]))
                    
                    diff =rep1-rep2
                    dis =np.dot(diff,diff)
                    eat_norm +=dis

                if eat_norm/len(base_dirout)<dis_thr:
                    for i in range(len(next_dirout)):
                        shutil.move(os.path.join(next_subdir_path,next_dirout[i]),base_subdir_path)
                        print('************************************************')
                        print("++++++++Moving {} to {}.".format(next_dirout[i],base_subdir_path))
                        print('************************************************')
                    os.removedirs(next_subdir_path)
                else:
                    pass

        num +=1
        dirout1=os.listdir(img_path)
        dirout1.sort()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("The remaining subdirs is {}.".format(len(dirout1)))
        if num==(len(dirout1)-1):
            break

    print("The final number of cluster is {}.".format(num+1))

