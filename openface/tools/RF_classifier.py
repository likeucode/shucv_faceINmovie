#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import itertools
import os
import pickle
import time

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import openface
import openface.helper
from openface.data import iterImgs

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(imgPath):
    img = cv2.imread(imgPath)

    if img is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    if args.verbose:
        print("  + Original size: {}".format(img.shape))

    bb = align.getLargestFaceBoundingBox(img)
    if bb is None:
        alignedFace=cv2.resize(img,(args.imgDim,args.imgDim),interpolation=cv2.INTER_AREA)

    alignedFace = align.alignImg("affine", args.imgDim, img, bb)
    if alignedFace is None:
        alignedFace=cv2.resize(img,(args.imgDim,args.imgDim),interpolation=cv2.INTER_AREA)



    rep = net.forwardImage(alignedFace)
    return rep


def train(args):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(args.workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    # clf = RandomForestClassifier(n_estimators=25)
    # clf.fit(embeddings,labelsNum)

    param_grid = [
        {'n_estimators': [10, 20,100,110,200,400,800],'criterion': ['gini']}
    ]
    rf = GridSearchCV(
        RandomForestClassifier(warm_start=False,min_samples_split = 1,n_jobs=4),
        param_grid, verbose=2, cv=5, n_jobs=16
    ).fit(embeddings, labelsNum)
    print("Best estimator: {}".format(rf.best_estimator_))
    print("Best score on left out data: {:.2f}".format(rf.best_score_))

    with open("{}/rf_classifier.pkl".format(args.workDir), 'w') as f:
        pickle.dump((le, rf), f)


def infer(imgpath):
    with open(args.classifierModel, 'r') as f:
        (le, rf) = pickle.load(f)
    rep = getRep(imgpath)
    predictions = rf.predict_proba(rep)[0]
    #print predictions
    maxI = np.argmax(predictions)
    max10I=np.argsort(predictions)[-10:]
    person = le.inverse_transform(maxI)
    person10=le.inverse_transform(max10I)
    confidence = predictions[maxI]
    confidence10=predictions[max10I]
    print("Predict {} with {:.2f} confidence.".format(person, confidence))
    #print("Top 10 {} with {} confidence.".format(person10, confidence10))
    return person,person10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dlibFaceMean', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "mean.csv"))
    parser.add_argument('--dlibFacePredictor', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir,
                                             "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--dlibRoot', type=str,
                        default=os.path.expanduser(
                            "~/src/dlib-18.16/python_examples"),
                        help="dlib directory with the dlib.so Python library.")
    parser.add_argument('--networkModel', type=str,
                        help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('workDir', type=str,
                             help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

    inferParser = subparsers.add_parser('infer',
                                        help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument('classifierModel', type=str)
    inferParser.add_argument('img', type=str,
                             help="Input image.")

    args = parser.parse_args()

    sys.path.append(args.dlibRoot)
    import dlib
    from openface.alignment import NaiveDlib  # Depends on dlib.

    align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
    net = openface.TorchWrap(
        args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        img_cnt=0
        top1_acc_cnt=0
        top10_acc_cnt=0
        identities=os.listdir(args.img) #it's a list
        numid=len(identities) #number of persons in the database
        print args.img,identities,type(identities)
        print ("number of persons in the database:",numid)
        for person in identities:
            personpath=os.path.join(args.img,person) #the path to the dictionary of each person
            items=os.listdir(personpath) #it's a list
            for item in items:
                img_cnt+=1
                print "-----------------------------------------------------------------------"
                print 'image index:',img_cnt
                itempath=os.path.join(personpath,item)
                [top1person,top10person]=infer(itempath)
                trueid=itempath.split("/")[-2]
                print "trueid:",trueid
                if top1person ==trueid:
                    top1_acc_cnt+=1
                if trueid in top10person:
                    top10_acc_cnt+=1                    
                else:
                    continue
                print 'top1_acc_cnt: ',top1_acc_cnt
                print 'top10_acc_cnt: ',top10_acc_cnt

        print ('total number of image: ',img_cnt)
        print ('top1_acc is: %f'%(float(top1_acc_cnt)/img_cnt))
        print ('top10_acc is: %f'%(float(top10_acc_cnt)/img_cnt))
        