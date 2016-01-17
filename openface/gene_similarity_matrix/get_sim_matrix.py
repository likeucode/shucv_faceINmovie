#!/usr/bin/python
#
# get similarity matrix
# liuke
# 2016/01/15

import argparse
import cv2
import itertools
import os

import numpy as np
# np.set_printoptions(precision=2)
from operator import itemgetter
import scipy as sp
from scipy import io as spio
import pandas as pd
import sys

def find_index(x,y):
    return [a for a in range(len(y)) if y[a]==x]

def cal_similarity(id,name,rep,num):
    simi_mat=np.zeros([rep.shape[0],rep.shape[0]])

    for i in range(rep.shape[0]):
        diff=rep[i]-rep
        simi_mat[i]=np.sum(diff**2,axis=1)
        print ("Processing number: ",i)

    print("save similarity mat file...")
    spio.savemat('s'+str(num+1)+'_matrix.mat',{'simi':simi_mat,'id':id,'name':name})

    # return simi_mat,ids,names

if __name__=='__main__':

    rep_path='./facenet_feature/S01E01'
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(rep_path)
    all_labels= pd.read_csv(fname, header=None).as_matrix()
    all_ids=all_labels[:,0]
    all_names=all_labels[:,1]
    all_names=map(itemgetter(1),map(os.path.split,all_names))
    for l in range(len(all_names)):
        all_names[l]=all_names[l][:-4]
    all_names=np.array(all_names)

    fname = "{}/reps.csv".format(rep_path)
    embeddings = pd.read_csv(fname, header=None).as_matrix()

    all_actors=list(set(all_ids))
    for ite in range(len(all_actors)):
    	idx=find_index(all_actors[ite],all_ids)
    	print type(all_ids),type(all_names),type(embeddings)
    	cal_similarity(all_ids[idx],all_names[idx],embeddings[idx],ite)

    print("Finished!")

    # simi_mat=np.zeros([embeddings.shape[0],embeddings.shape[0]])

    # for i in range(embeddings.shape[0]):
    #     diff=embeddings[i]-embeddings
    #     simi_mat[i]=np.sum(diff**2,axis=1)
    #     print ("Processing number: ",i)
    # print("save similarity mat file...")
    # spio.savemat('simi_matrix.mat',{'simi':simi_mat,'ids':ids,'names':names})
    # print("Finished!")
