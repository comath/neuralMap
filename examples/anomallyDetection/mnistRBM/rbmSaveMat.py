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
for i in range(5):
	matricies =  {}
	for hiddenDim in range(10,101,5):
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
		
		for k in range(1, 10001):
			tr_x, tr_y  = mnist.train.next_batch(batchSize)
			sess.run(train, feed_dict={X: tr_x})
			

		matricies["matrix%(hidden)04d"% {'hidden': hiddenDim}] = sess.run(A)
		matricies["offsetVis%(hidden)04d"% {'hidden': hiddenDim}] = sess.run(bvis)
		matricies["offsetHid%(hidden)04d"% {'hidden': hiddenDim}] = sess.run(bhid)

	io.savemat("mnistRBM%(itter)d.mat"%{'itter':i},matricies)