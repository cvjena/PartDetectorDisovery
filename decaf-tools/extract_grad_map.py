# Load the data. Make sure you have the right path to the jeffnet model file.
# Note: due to the training scripts, everything we visualize here is upside-down. Bear with us.

from matplotlib import pyplot
import numpy as np
import pylab
import pprint
import scipy.misc
from subprocess import call
from decaf.scripts import jeffnet
from decaf.util import smalldata
from decaf.util import visualize
import os
import thread
import time
from multiprocessing import Process, Pool
import re
import scipy.io as sio
from scipy.misc import imresize
import time
import argparse


# call(["./build_decaf"])

def normalize(array):
    array-=np.min(array)
    array/=np.max(array)
    return array

def calcMap(net,gradient_layer,channel_idx,limit):
    #print "calc map config: layer="+`gradient_layer`+",channel_idx="+`channel_idx`+",limit="+`limit`
    #pyplot.figure("layer="+`gradient_layer`+",channel_idx="+`channel_idx`+",limit="+`limit`);
    (gradient,channel) = net._net.forward_backward(gradient_layer=gradient_layer,limit=limit, channel=channel_idx)
    gradient_norm = np.linalg.norm(gradient,ord=2,axis=(3))
    #gradient_norm = np.tensordot(probs[:gradient.shape[0]],gradient_norm,axes=0);
    #gradient_norm = np.tensordot(np.ones(gradient.shape[0]),gradient_norm,axes=0);
    # if the probs layer is selected, weight the gradient maps by probability
    if gradient_layer=="probs":
        probs=-np.sort(-net._net._backward_order[0][3][0]._data.copy(), axis=1)[0][:limit]
    else:
        probs=np.ones_like((gradient_norm[:,0,0]))
    tmp = np.zeros(gradient_norm.shape[1:])
    for j in range (gradient.shape[0]):
        tmp += gradient_norm[j,:,:]*probs[j]
    gradient_norm = np.flipud(tmp)
    #gradient_norm = normalize(gradient_norm)
    # The layers available
    #datablob=bottom[0]._diff[0]
    #outmap=np.flipud(np.amax(np.absolute(datablob),2))
#     #outmap=np.flipud(np.linalg.norm(datablob,ord=2,axis=2))
#     outmap=np.tile(normalize(np.flipud(gradient_norm))[:,:,None],(1,1,3));
#     tmp=image.copy()
#     tmp*=outmap;
#     visualize.show_single(normalize(tmp.copy()))
#     pyplot.savefig("hund_layer"+`gradient_layer`+"_channel"+`i`)
#     pyplot.figure("map"+`i`)
#     visualize.show_single(normalize(np.flipud(gradient_norm)))
#     pyplot.savefig("hund_map"+"_layer"+`gradient_layer`+"_channel"+`i`)
    return (gradient_norm,channel)

def addPicToHtml(array,dir,imagename,outfile,title):
    scipy.misc.imsave(dir+"/"+imagename, array)
    outfile.write("<div>"+title+"<br><img src='"+imagename+"'></div>")


def extractFromImage(imagepath,outdir,config):
    print "Start working on image "+imagepath
    #print "Opening html file..."
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outdir+"/gradient.html", 'a') as html:
        html.write("<style>div {    display:block;    margin-left:50%;    width:40%;    text-align:center;}div:nth-child(1) {    margin-left:0%;    position:fixed;    top:10%;    left:5%;}div:nth-child(2) {    margin-top:10%;}img {    width:100%;}</style>\n")    
        print "Initializing Neural Network..."
        data_root='models/'
        net = jeffnet.JeffNet(data_root+'imagenet.decafnet.epoch90', data_root+'imagenet.decafnet.meta')
    
        # Run a classification pass to create all the intermediate features
        #print "Classifying image..."
        img = pylab.imread(imagepath)
        img = imresize(img,(256,256,3),interp='bicubic')
        #print "Resizing image..."
        #pyplot.imshow(img)
        #pyplot.show()
        #print img.shape
        net.classifiyWholeImage(img)
        t = time.time()
        image_normalized=normalize(np.flipud(net._net._forward_order[0][2][0]._data[0]).copy())
        addPicToHtml(image_normalized,outdir,"inputimage.jpg", html,'Eingabebild')
        sio.savemat(outdir+"/inputimage.mat", {'image_normalized':image_normalized},do_compression=True)
    
        for gradient_layer in config.layers:
            if gradient_layer=="probs":  
                print "Calculating probability gradient..."
                (gradient_map,_)=calcMap(net,"probs",0,config.limit)
                addPicToHtml(gradient_map,outdir,"gradient_probs.png", html, 'Probability Gradient Map')
                sio.savemat(outdir+"/gradient_probs.mat", {'gradient_map':gradient_map},do_compression=True)
            else:
                for i in range(256):#[5,53,72,170,207]:#
                    #if not os.path.isfile(outdir+"/gradient"+"_layer"+gradient_layer+"_channel"+`i`+'.mat') or not os.path.isfile(outdir+"/gradient"+"_layer"+gradient_layer+"_channel"+`i`+'.png'):
                    #print "Calculating gradient for layer "+`gradient_layer`+", channel "+`i`
                    (gradient_map,channel)=calcMap(net,gradient_layer,i,1)
                    addPicToHtml(gradient_map, outdir,"gradient"+"_layer"+gradient_layer+"_channel"+`channel`+'.png',html, 'Gradient for layer '+`gradient_layer`+', '+`i`+'th channel '+`channel`)
                    sio.savemat(outdir+"/gradient"+"_layer"+gradient_layer+"_channel"+`channel`+'.mat', {'gradient_map':gradient_map},do_compression=True)
	# do stuff
	elapsed = time.time() - t
	print elapsed
        #print "Finished!"

parser = argparse.ArgumentParser(description="Calculate the gradient maps for an image.")
parser.add_argument('--layers', nargs='+',default=["probs","pool5"])
parser.add_argument('--limit', type=int, default=10,help="When calculating the gradient of the probability, calculate the gradient for the channels with the [limit] highest probabilities.")
parser.add_argument('--channel_limit', type=int, default=256,help="Sets the number of channels per layer you want to calculate the gradient of.")
parser.add_argument('--images',metavar='pattern',nargs='+', default=["/home/ubuntu/FineGrained/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0049_796063.jpg"])
parser.add_argument('--outdir', action='store',default="out")
config = parser.parse_args()

# Create two threads as follows
pool = Pool(processes=4)   
res = []
for i,ipath in enumerate(config.images):
    res.append(pool.apply_async(extractFromImage, [config.images[i],config.outdir+"/"+re.search('.*/([^/]+/[^/]+)', config.images[i], re.IGNORECASE).group(1),config ]))
    #extractFromImage(config.images[i],config.outdir+"/"+re.search('.*/([^/]+/[^/]+)', config.images[i], re.IGNORECASE).group(1),config)
for r in res:
    r.get()