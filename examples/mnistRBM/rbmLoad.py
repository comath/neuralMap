import sqlite3
import tensorflow as tf
import numpy as np
from scipy import io
from rbm import RBM
from PIL import Image

from nnMapper import nnMapper as nnMap
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
currentNumLocations = 0
currentNumPoints = 0

print trX.shape

for hiddenDim in range(20,21,20):
	matDic = {}
	io.loadmat("mnist%(visibleDim)dx%(hiddenDim)03dstepsize%(stepSize)f.mat" 
													% {'visibleDim': visibleDim, 'hiddenDim': hiddenDim, 'stepSize':stepSize},
													matDic)
	
	matrix = matDic["matrix%(round)04d"% {'round': itte}]
	offset = matDic["offsetVis%(round)04d"% {'round': itte}]
	matrix = np.ascontiguousarray(matrix.T, dtype=np.float32)
	print matrix.shape
	offset = np.ascontiguousarray(offset, dtype=np.float32)
	offset.shape = offset.shape[1]
	map1 = nnMap(matrix,offset,'mnistRBM.db','hidden%(hid)03d' % {'hid':hiddenDim})
	indicies = range(100)
	map1.addPoints(indicies,trX[0:100,])
#	for k in range(1, 60):
		
		

