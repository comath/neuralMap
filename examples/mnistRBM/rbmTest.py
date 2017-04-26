import sqlite3
import numpy as np
from scipy import io
from rbm import RBM
from PIL import Image

from nnMapper import nnMapper as nnMap
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
mnist.test.images, mnist.test.labels

leaveInData = []
leaveInLabel = []
removeData = []
removeLabel = []

for i,label in enumerate(teY):
	if(label[9]==1 or label[8] == 1):
		removeData.append(teX[i])
		removeLabel.append(label)
	else:
		leaveInData.append(teX[i])
		leaveInLabel.append(label)
leaveInData = np.stack(leaveInData,axis=0)
leaveInLabel = np.stack(leaveInLabel,axis=0)
removeData = np.stack(removeData,axis=0)
removeLabel = np.stack(removeLabel,axis=0)

testCount = 400

indicies = range(0,testCount)
leaveInData = leaveInData[0:testCount,]
removeData = removeData[0:testCount,]

visibleDim = 28*28
batchSize = 1000
stepSize = 0.005
itte = 10000

for hiddenDim in range(20,101,20):
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
	
	leaveCount = 0.0
	removeCount = 0.0
	pointTest1 = map1.checkPoints(indicies,leaveInData)
	pointTest2 = map1.checkPoints(indicies,removeData)
	for check in pointTest1:
		if(check[1]):
			leaveCount +=1
	for check in pointTest2:
		if(check[1]):
			removeCount +=1
	print("The success rate of the leaveIn classes is: %(suc)f" % {'suc':leaveCount/leaveInData.shape[0]})
	print("The success rate of the remove classes is: %(suc)f" % {'suc':removeCount/removeData.shape[0]})