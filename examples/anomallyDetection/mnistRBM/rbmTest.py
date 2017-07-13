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

visibleDim = 28*28
batchSize = 1000
stepSize = 0.005
itte = 10000


for hiddenDim in range(10,101,5):
	matDic = {}
	io.loadmat("mnist%(visibleDim)dx%(hiddenDim)03dstepsize%(stepSize)f.mat" 
						% {'visibleDim': visibleDim, 'hiddenDim': hiddenDim, 'stepSize':stepSize},
						matDic)
	
	matrix = matDic["matrix%(round)04d"% {'round': itte}]
	offset = matDic["offsetVis%(round)04d"% {'round': itte}]
	offset.shape = offset.shape[1]
	map1 = nnMap([matrix],[offset])
	map1.load('mnistRBM.db',table_name='hidden%(hid)03d' % {'hid':hiddenDim})
	
	leaveCount = 0.0
	removeCount = 0.0
	pointTest1 = map1.check(leaveInData,reg_only=True)
	pointTest2 = map1.check(removeData,reg_only=True)
	pointTest1both = map1.check(leaveInData)
	pointTest2both = map1.check(removeData)
	for check in pointTest1:
		if(check):
			leaveCount +=1
	for check in pointTest2:
		if(not check):
			removeCount +=1
	print("For %(hid)d nodes:"%{'hid':hiddenDim})
	print("The success rate of the leaveIn classes is (percent of non-anomalies recognised as such): %(suc)f" % {'suc':leaveCount/leaveInData.shape[0]})
	print("The success rate of the remove classes is (percent of anomalies recognised as such): %(suc)f" % {'suc':removeCount/removeData.shape[0]})

	leaveCountboth = 0.0
	removeCountboth = 0.0
	for check in pointTest1both:
		if(check):
			leaveCountboth +=1
	for check in pointTest2both:
		if(not check):
			removeCountboth +=1
	print("Using both IP and Reg, the success rate of the leaveIn classes is (percent of non-anomalies recognised as such): %(suc)f" % {'suc':leaveCountboth/leaveInData.shape[0]})
	print("Using both IP and Reg, the success rate of the remove classes is (percent of anomalies recognised as such): %(suc)f" % {'suc':removeCountboth/removeData.shape[0]})