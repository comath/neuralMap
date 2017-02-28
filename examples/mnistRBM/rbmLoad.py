import sqlite3
import tensorflow as tf
import numpy as np
from scipy import io
from rbm import RBM
from PIL import Image

from mapperWrap import nnMap
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
mnist.test.images, mnist.test.labels

visibleDim = 28*28
batchSize = 1000
stepSize = 0.005
itte = 10000

errorMargins = np.zeros([batchSize],dtype=np.float32)

def convertToString(ndarr):
	retString = ''
	for i in range(ndarr.shape[0]):
		retString = retString + "%(k)d" % {'k':ndarr[i]}
	return retString

for hiddenDim in range(100,101,20):
	matDic = {}
	io.loadmat("mnist%(visibleDim)dx%(hiddenDim)03dstepsize%(stepSize)f.mat" 
													% {'visibleDim': visibleDim, 'hiddenDim': hiddenDim, 'stepSize':stepSize},
													matDic)
	
	matrix = matDic["matrix%(round)04d"% {'round': itte}]
	offset = matDic["offsetVis%(round)04d"% {'round': itte}]
	matrix = np.ascontiguousarray(matrix, dtype=np.float32)
	print matrix.shape
	offset = np.ascontiguousarray(offset, dtype=np.float32)
	offset.shape = offset.shape[1]
	hidMat = np.zeros([hiddenDim,2],dtype=np.float32)
	hidOff = np.zeros([2],dtype=np.float32)
	map1 = nnMap(matrix,offset,hidMat,hidOff,2,0.5)
	for k in range(1, 60):
		tr_x, tr_y  = mnist.train.next_batch(batchSize)
		map1.batchAdd(tr_x,errorMargins,1)
		

	for i in range(map1.numLocations()):
		curloc = map1.location(i)
		ip = curloc.ipSig()
		reg = curloc.regSig()
		avgPoint = curloc.avgPoint()
		avgPoint.shape = [28,28]
		ipString = convertToString(ip)
		regString = convertToString(reg)

		image = Image.fromarray(avgPoint)
		if image.mode != 'RGB':
			image = image.convert('RGB')
		image.save("ip" + ipString + "reg" + regString + ".png")
