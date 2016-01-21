#!/usr/bin/env python2
#
# compare the faces in two images and make database.
# liuke
# 2016/01/21
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
        #raise Exception("Unable to align image: {}".format(imgPath))
        print("Unable to align image: {}".format(imgPath))
        os.remove(imgPath)
        flag=False
        return flag,rep
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
    actor_root='/home/liuke/master/master2016/openface/kelly/actors/'
    actor_list=actor_root+'list.txt' #find `pwd`/path/to/image/root -name *.jpg >./list.txt
    actor_path=np.loadtxt(actor_list,str,delimiter='\n')
    
    img_path='/home/liuke/master/master2016/openface/kelly/wow/'
    dirout1=os.listdir(img_path)
    actor_num=0
    dis_thr=0.4
    '''
    for i in range(len(actor_path)):
    	img1=actor_path[i]
    	flag1,rep1=getRep(img1)
        target_img=img1.split('/')[-1]
        actor=img1.split('/')[-2]
        des_dir=os.path.join(actor_root,actor)

        if not flag1:
            continue
        else:
            pass
           

        dirout2=os.listdir(img_path)
        for img2 in dirout2:
            flag2,rep2=getRep(img_path+img2)
            if not flag2:
                continue
            else:
                pass

            d=rep1-rep2
            dl2=np.dot(d,d)
            if dl2<dis_thr:
                shutil.move(img_path+img2,des_dir)
            else:
                pass
            print("Processing number of actor {},total imgs is {}".format((i+1),len(dirout2)))
            print("Comparing {} with {}.".format(target_img, img2))
            print("  + Squared l2 distance between representations: {:0.3f}".format(dl2))
    '''
    probe_dirout=os.listdir(img_path)
    for probe in probe_dirout:
        probe_flag,probe_rep=getRep(img_path+probe)
        if not probe_flag:
            continue
        else:
            pass

        tmp_dis=np.array([0.01])
        
        for j in range(len(actor_path)):
            gallery=actor_path[j]
            gallery_img=gallery.split('/')[-1]
            
            gallery_flag,gallery_rep=getRep(gallery)
            diff=probe_rep-gallery_rep
            dispg=np.dot(diff,diff)
            tmp_dis=np.append(tmp_dis,dispg)
            print("Comparing {} with {}.".format(probe, gallery_img))
            print("  + Squared l2 distance between representations: {:0.3f}".format(dispg))

        final_dis=tmp_dis[1:]       
        target_idx=np.argmin(final_dis)
        actor_des=actor_path[target_idx].split('/')[-2]
        target_des=os.path.join(actor_root,actor_des)
        shutil.move(img_path+probe,target_des)




    