import sqlite3
import numpy as np
from scipy import io
from rbm import RBM
from PIL import Image

from nnMap import nnMap
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
mnist.test.images, mnist.test.labels

leaveInData = []
leaveInLabel = []
removeData = []
removeLabel = []

for i,label in enumerate(trY):
	if(label[9]==1 or label[8] == 1):
		removeData.append(trX[i])
		removeLabel.append(label)
	else:
		leaveInData.append(trX[i])
		leaveInLabel.append(label)
leaveInData = np.stack(leaveInData,axis=0)
leaveInLabel = np.stack(leaveInLabel,axis=0)
removeData = np.stack(removeData,axis=0)
removeLabel = np.stack(removeLabel,axis=0)

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

for i in range(5):
	matDic = {}
	io.loadmat("mnistRBM%(itter)d.mat"%{'itter':i},	matDic)
	for hiddenDim in range(10,101,5):		
		matrix = matDic["matrix%(hidden)04d"% {'hidden': hiddenDim}]
		offset = matDic["offsetVis%(hidden)04d"% {'hidden': hiddenDim}]
		offset.shape = offset.shape[1]
		map1 = nnMap([matrix],[offset])
		indicies = np.arange(leaveInData.shape[0],dtype=np.int32)
		map1.add(leaveInData,indicies)
		map1.save('mnistRBM.db',table_name='hidden%(hid)03ditte%(itter)d' % {'hid':hiddenDim,'itter':i})


