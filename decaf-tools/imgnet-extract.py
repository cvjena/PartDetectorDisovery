import argparse
import sys
import numpy as np
import matplotlib.pyplot as pylab
from decaf.scripts.jeffnet import JeffNet
from decaf.util import smalldata
import csv
import os
import errno
import glob
from scipy.misc import imresize
import scipy.io as sio


def writeFeatures(net,img_path,all_features,layer,csv_writer,silent,warp,mat_path):
  if not silent:
    print img_path
  if img_path=='0':
    csv_writer.writerow([0] * 4096)
    return
  img = pylab.imread(img_path)
  if warp:
    img = imresize(img,(256,256,3),interp='bicubic')
    scores = net.classifiyWholeImage(img)
  elif all_features:
    scores = net.classify(img)
  else:
    scores = net.classify(img,center_only=True)
  fv= net.feature(args.layer)[0]
  data_blob= net.feature('data')[0]
  sio.savemat(mat_path,{'out':fv,'img':data_blob})
  csv_writer.writerow(fv.reshape((1,-1))[0])


data_root='/home/simon/Research/lib/DecafDev/deep-model/models/'

parser = argparse.ArgumentParser(description='Calculate the features of the given input images using a convolutional network. ')
parser.add_argument('img_path_patterns',metavar='pattern', nargs='+',action='append',help='A path pattern for the input images. For example, ./lena.jpg will select lena.jpg from the current directory and ./*.jpg will select all images with the extension jpg in the current directory. If --file-list-mode is set, pattern refers to a path to a text file containing the paths to the images.')
parser.add_argument('--model',nargs='?',default=data_root+'imagenet.jeffnet.epoch90',action='store',help='Path to a model of a convulutional network.')
parser.add_argument('--meta',nargs='?',default=data_root+'imagenet.jeffnet.meta',action='store',help='Path to the meta data of the convulutional network.')
parser.add_argument('--out',nargs='?',default='feature_out.csv',action='store',help='File to store the DeCAF features. The output is a comma separated values (CSV) file. Each line corresponds to one input image.')
parser.add_argument('--layer',nargs='?',default='fc7_neuron_cudanet_out',action='store',help='The features of the layer specified here will be used.')
parser.add_argument('--all-features',action='store_true',help='Wheather to store all features or just for the center part of the image.')
parser.add_argument('--file-list-mode',action='store_true',help='If you set this flag, you can use a file list as input.')
parser.add_argument('--silent',action='store_true',help='Suppress output to console.')
parser.add_argument('--warp',action='store_true',help='Warp the input image to 227x227 and use this directly as input.')


args = parser.parse_args()
# Early check, if an image path is passed
args.img_path_patterns

net = JeffNet(args.model, args.meta)

csv_file=open(args.out,"w");
csv_writer = csv.writer(csv_file,delimiter=',',  quoting=csv.QUOTE_NONE) 

for (i,img_list) in enumerate(args.img_path_patterns):
  for img_path in img_list:
    if args.file_list_mode:
      with open(img_path) as f:
        content = f.readlines()
        for img_name in content:
          writeFeatures(net,img_name.strip('\n'),args.all_features,args.layer,csv_writer,args.silent,args.warp,args.out+'.mat')
    else:
      writeFeatures(net,img_path,args.all_features,args.layer,csv_writer,args.silent,args.warp,args.out+'.mat')

