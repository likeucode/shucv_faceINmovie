import numpy as np
from scipy import io as spio
import os
from operator import itemgetter
import sys
caffe_root='../'
sys.path.insert(0,caffe_root+'python')

import caffe

caffe.set_mode_cpu()
img_mean=caffe_root+'vgg_face_caffe/vgg_ytb_mean.npy'
model_conf=caffe_root+'vgg_face_caffe/VGG_FACE_deploy.prototxt'
model_weights=caffe_root+'vgg_face_caffe/VGG_FACE.caffemodel'

net=caffe.Net(model_conf,model_weights,caffe.TEST)

transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',np.load(img_mean))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))

net.blobs['data'].reshape(50,3,224,224)

img_list=caffe_root+'vgg_face_caffe/list.txt' #find `pwd`/path/to/image/root -name *.jpg >./list.txt
img_path=np.loadtxt(img_list,str,delimiter='\n')
vgg_feat=np.zeros([1,4096])
labels=np.array([''])
for i in range(len(img_path)):
    img=img_path[i]
    net.blobs['data'].data[...]=transformer.preprocess('data',caffe.io.load_image(img))
    out=net.forward()
    feat=net.blobs['fc7'].data[0].reshape((1,4096))
    vgg_feat=np.append(vgg_feat,feat,axis=0)

    tmp_label=img.split('/')[-2]
    labels=np.append(labels,tmp_label)
    # print("Predicted class is #{}.".format(out['prob'][0].argmax()))

vgg_feat=vgg_feat[1:]
labels=labels[1:]
spio.savemat('vgg_face.mat',{'feat':vgg_feat,'labels':labels})



