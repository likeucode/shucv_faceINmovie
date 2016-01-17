import numpy as np
import os
import sys
caffe_root='../'
sys.path.insert(0,caffe_root+'python')

import caffe

caffe.set_mode_cpu()
img_path=caffe_root+'./vgg_face_caffe/Colin_Donnell408.jpg'
img_mean=caffe_root+'./vgg_face_caffe/vgg_ytb_mean.npy'
model_conf=caffe_root+'./vgg_face_caffe/VGG_FACE_deploy.prototxt'
model_weights=caffe_root+'./vgg_face_caffe/VGG_FACE.caffemodel'

net=caffe.Net(model_conf,model_weights,caffe.TEST)

transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',np.load(img_mean))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))

net.blobs['data'].reshape(50,3,224,224)
net.blobs['data'].data[...]=transformer.preprocess('data',caffe.io.load_image(img_path))
out=net.forward()
print("Predicted class is #{}.".format(out['prob'][0].argmax()))

