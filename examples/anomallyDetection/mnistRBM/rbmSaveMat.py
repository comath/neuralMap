import sqlite3
import tensorflow as tf
import numpy as np
from scipy import io


from rbm import RBM

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
batchSize = 100
stepSize = 0.005

for hiddenDim in range(20,101,20):
	tf.reset_default_graph()
	testRBM = RBM(visibleDim,hiddenDim)
	train = testRBM.contrastiveDivergenceN(1,stepSize)
	X,Y = testRBM.placeholders()
	A = testRBM.getWeightsPointer()
	bvis = testRBM.getVisibleBiasPointer()
	bhid = testRBM.getHiddenBiasPointer()
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	tr_x, tr_y  = mnist.train.next_batch(batchSize)
	mse = testRBM.mse(tf.cast(tr_x, tf.float32))
	matricies =  {}
	for k in range(1, 10001):
		tr_x, tr_y  = mnist.train.next_batch(batchSize)
		sess.run(train, feed_dict={X: tr_x})
		
		if(k%100 == 0 ):
			matricies["matrix%(batch)04d"% {'batch': k}] = sess.run(A)
			matricies["offsetVis%(batch)04d"% {'batch': k}] = sess.run(bvis)
			matricies["offsetHid%(batch)04d"% {'batch': k}] = sess.run(bhid)
			
	
	io.savemat("mnist%(visibleDim)dx%(hiddenDim)03dstepsize%(stepSize)f.mat" 
													% {'visibleDim': visibleDim, 'hiddenDim': hiddenDim, 'stepSize':stepSize},
													matricies)

