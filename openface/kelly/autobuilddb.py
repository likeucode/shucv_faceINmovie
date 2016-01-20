#!/usr/bin/env python2
#
# compare the faces in two images and make database.
# liuke
# 2016/01/20
#

import time

start = time.time()

import argparse
import cv2
import itertools
import os
import shutil

import numpy as np
np.set_printoptions(precision=2)
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import openface

#fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
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


def getRep(imgPath):
    flag=True
    rep=0
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        #raise Exception("Unable to load image: {}".format(imgPath))
        print("Unable to load image: {}".format(imgPath))
        os.remove(imgPath)
        flag=False
        return flag,rep
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        #raise Exception("Unable to find a face: {}".format(imgPath))
        print("Unable to find a face: {}".format(imgPath))
        os.remove(imgPath)
        flag=False
        return flag,rep
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return flag,rep

for (img1, img2) in itertools.combinations(args.imgs, 2):
    flag,rep=getRep(img1)
    if not flag:
        continue
    else:
        d1=rep
    flag,rep=getRep(img2)
    if not flag:
        continue
    else:
        d2=rep
    d=d1-d2
    dl2=np.dot(d,d)
    if dl2<1.0:
        shutil.move(img2,'./data/')
    else:
        pass
    print("Comparing {} with {}.".format(img1, img2))
    print("  + Squared l2 distance between representations: {:0.3f}".format(dl2))
