#!/usr/bin/env python2
'''
This is a demo for mid-term project
author:liuke
date:2015/12/25
version:v1.2
'''

import argparse
import cv2
import itertools
import os
import pickle
import os
import time
import random

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
import scipy as sp

import dlib
detector = dlib.get_frontal_face_detector()

import sys
openface_root="/home/byx/openface-master/"
sys.path.insert(0,openface_root)

import openface
import openface.helper
from openface.data import iterImgs
from openface.alignment import NaiveDlib  # Depends on dlib.

modelDir = os.path.join(openface_root, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')



from sklearn.metrics.pairwise import euclidean_distances

def find_index(x,y):
    return [a for a in range(len(y)) if y[a]==x]

def load_db(db_path):
    print('loading main database...')
    db=np.load(db_path)
    return db

def getRep(img_path):
    img=cv2.imread(img_path)
    '''
    bb = align.getLargestFaceBoundingBox(img)
    if bb is None: #find no face in the img, maybe due to large face.
        #raise Exception("Unable to find a face: {}".format(imgPath))
        alignedFace=cv2.resize(img,(args.imgDim,args.imgDim),interpolation=cv2.INTER_AREA)
    else:
        img_crop = img[bb.top():bb.bottom(),bb.left():bb.right(),:]
        #cv2.imshow(img_crop)
    	#cv2.waitKey(0)
        s=img_crop.shape
        if s[0]==0 or s[1]==0:
        	alignedFace=cv2.resize(img,(args.imgDim,args.imgDim),interpolation=cv2.INTER_AREA)
        else:
        	alignedFace=cv2.resize(img_crop,(args.imgDim,args.imgDim))
    '''
    rep = net.forwardImage(img)
    rep=rep.reshape((128,1))
    return rep

def compare_db(img_rep,subdb_rep):
    print('Calculating l2 distance...')
    dis=euclidean_distances(img_rep.T,subdb_rep.T)
    #get index
    index=np.argsort(dis[0,:])
    #sort
    sorted_dis=np.sort(dis[0,:])
    top1_index=index[0]
    top10_index=index[0:10]

    return top1_index,top10_index

def get_2example_subdb(db):
    print('building subdb')
    dbrep=db['arr_0']
    dbid=db['arr_1']
    subdb_rep=np.zeros([128,1])
    subdb_ids=np.array([])
    #people=set(dbid)
    people={'E00001','E00003'}

    for id in people:
        idx=find_index(id,dbid)
        if len(idx)<10:
            continue
        else:
            sample_rep_idx=random.sample(idx,10)
            sample_idx=random.sample(dbid[idx],10)
            subdb_ids=np.append(subdb_ids,sample_idx)
            subdb_rep=np.append(subdb_rep,dbrep[:,sample_rep_idx],axis=1)

    subdb_rep=subdb_rep[:,1:]

    sub_db={'data':subdb_rep,'ids':subdb_ids}
    print subdb_ids,subdb_ids.shape
    return sub_db


def get_subdb(db):
    print('building subdb')
    dbrep=db['arr_0']
    dbid=db['arr_1']
    subdb_rep=np.zeros([128,1])
    subdb_ids=np.array([])
    people=set(dbid)

    for id in people:
        idx=find_index(id,dbid)
        if len(idx)<10:
            continue
        else:
            sample_rep_idx=random.sample(idx,10)
            sample_idx=random.sample(dbid[idx],10)
            subdb_ids=np.append(subdb_ids,sample_idx)
            subdb_rep=np.append(subdb_rep,dbrep[:,sample_rep_idx],axis=1)

    subdb_rep=subdb_rep[:,1:]

    sub_db={'data':subdb_rep,'ids':subdb_ids}
    print subdb_ids,subdb_ids.shape
    return sub_db


def vali_recall_acc(test_path,sub_db):
    subdb_rep=sub_db['data']
    subdb_ids=sub_db['ids']

    test_src=os.listdir(test_path)
    test_size=len(test_src)
    print ('%d people will be tested'%test_size)
    top1_acc_cnt=0
    top10_acc_cnt=0
    top10_recall_cnt=0
    test_num=0
    img_cnt=0
    for person in test_src:
        person_path=os.path.join(test_path,person)
        person_list=os.listdir(person_path)
        test_num+=len(person_list)
        #print 'current person: ',person

        for img in person_list:
            if img[-3:]=='png':
                print 'img: ',img
                img_cnt+=1
                print 'number of image: ',img_cnt
                rep=getRep(os.path.join(person_path,img))
                [top1_index,top10_index]=compare_db(rep,subdb_rep)
            
                if person in subdb_ids[top1_index]:
                    top1_acc_cnt+=1
                    print 'top1_acc_cnt: ',top1_acc_cnt
                    print 'current person: ',person
                    print('current top1 is: ',subdb_ids[top1_index])

                if person in subdb_ids[top10_index]:
                    top10_acc_cnt+=1
                    print 'top10_acc_cnt: ',top10_acc_cnt
                    top10_recall_cnt+=list(subdb_ids[top10_index]).count(person)
                    print 'top10_recall_cnt: ',top10_recall_cnt
            else:
                continue
    print 'total number of image: ',img_cnt
    print "top1_acc is: %f"%(float(top1_acc_cnt)/img_cnt)
    print "top10_acc is: %f"%(float(top10_acc_cnt0/img_cnt)
    print "top10_recall is: %f"%(float(top10_recall_cnt)/(10*img_cnt))

if __name__=='__main__':
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

    net = openface.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)
    align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)

    db_path='./dbinfo_alignedtrain.npz'
    test_path='/home/byx/openface-master/test_acc/db/alignedtrain/'

    db=load_db(db_path)
    subdb=get_subdb(db)

    vali_recall_acc(test_path,subdb)


