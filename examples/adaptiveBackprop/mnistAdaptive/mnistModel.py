import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mapperWrap import nnMap
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
mnist.test.images, mnist.test.labels

inDim = 28*28
hiddenDim = currentHiddenDim = 15
maxHiddenDim = 25
outDim = 10
batchSize = 300

X = tf.placeholder(tf.float32,[None,inDim],name='Input')
Y = tf.placeholder(tf.int32,[None,outDim], name='Labels')

weights1 = {}
bias1 = {}
eval1 = {}
weights2 = {}
bias2 = {}
eval2 = {}
loss = {}
misclass = {}
accuracy = {}


for hd in range(currentHiddenDim,maxHiddenDim+1):
	with tf.name_scope('NeuralNetwork%(hidDim)d'%{'hidDim':hd}):
		with tf.name_scope('Layer1'):
			weights1[hd] = tf.Variable(tf.random_uniform([inDim,hd], -0.005, 0.005),name='Weights')
			bias1[hd] = tf.Variable(tf.random_uniform([hd], -0.005, 0.005),name='Bias')
			eval1[hd] = tf.nn.relu(tf.matmul(X,weights1[hd]) + bias1[hd],name='Evaluation')
		with tf.name_scope('Layer2'):
			weights2[hd] = tf.Variable(tf.random_uniform([hd,outDim], -0.005, 0.005),name='Weights',dtype=tf.float32)
			bias2[hd] = tf.Variable(tf.random_uniform([outDim], -0.005, 0.005),name='Bias',dtype=tf.float32)
			curEval2 = tf.nn.relu(tf.matmul(eval1[hd],weights2[hd]) + bias2[hd],name='Evaluation')
			eval2[hd] = curEval2

		loss[hd] = tf.losses.softmax_cross_entropy(Y,curEval2)
		#Summary Section
		with tf.name_scope('total'):
		    crossEntropy = tf.reduce_mean(loss[hd])
		softmaxLoss = tf.summary.scalar('softmax_loss%(hidDim)d'%{'hidDim':hd}, crossEntropy)

		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_prediction'):
				correct_prediction = tf.equal(tf.argmax(eval2[hd], 1), tf.argmax(Y, 1))
			with tf.name_scope('correct_prediction'):
				incorrect_prediction = tf.not_equal(tf.argmax(eval2[hd], 1), tf.argmax(Y, 1))
			with tf.name_scope('misclassifications'):
				misclass[hd] = tf.cast(incorrect_prediction, tf.int32)
			with tf.name_scope('accuracy'):
				accuracy[hd] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy%(hidDim)d'%{'hidDim':hd}, accuracy[hd])

merged = tf.summary.merge_all()

optimizer = tf.train.GradientDescentOptimizer(0.05)
globalStep = tf.Variable(0, name='globalStep', trainable=False)
trainOp = {}

for hd in range(currentHiddenDim,maxHiddenDim+1):
	trainOp[hd] = optimizer.minimize(loss[hd], global_step=globalStep, var_list=[weights1[hd],weights2[hd],bias1[hd],bias2[hd]])

sess = tf.Session()
writer = tf.summary.FileWriter('tfLogs/adaptiveStart%(hidDim)dEnd%(maxDim)d'%{'hidDim':hiddenDim,'maxDim':maxHiddenDim}, sess.graph)
init = tf.global_variables_initializer()
sess.run(init)


for i in range(10000):
	tr_x, tr_y = mnist.train.next_batch(batchSize)
	feedDict = {X:tr_x, Y:tr_y}
	summ,lossVal,trainVal = sess.run([merged,loss[currentHiddenDim],trainOp[currentHiddenDim]], feed_dict = feedDict)
	writer.add_summary(summ, i)

	if(i % 2000 == 0 and i>0 and currentHiddenDim < maxHiddenDim):
		

		npWeights1, npBias1, npWeights2,npBias2 = sess.run([weights1[currentHiddenDim],bias1[currentHiddenDim],weights2[currentHiddenDim],bias2[currentHiddenDim]])

		npWeights1 = np.ascontiguousarray(npWeights1.T, dtype=np.float32)
		npBias1 = np.ascontiguousarray(npBias1, dtype=np.float32)
		npWeights2 = np.ascontiguousarray(npWeights2.T, dtype=np.float32)
		npBias2 = np.ascontiguousarray(npBias2, dtype=np.float32)


		errors = sess.run([misclass[currentHiddenDim]], feed_dict = {X:trX, Y:trY})
		errors = np.concatenate(errors)

		neuralMap = nnMap(npWeights1,npBias1,2)
		indicies = np.arange(trX.shape[0],dtype=np.int32)
		neuralMap.batchAdd(trX,indicies,errors)
		newHiddenDim, npNewHPVec, npNewHPbias, npNewSelectionWeight, npNewSelectionBias = neuralMap.adaptiveStep(trX,npWeights2,npBias2)

		if(newHiddenDim == currentHiddenDim + 1):
			print("New Hyperplane created, inserting")
			currentHiddenDim = newHiddenDim
			newWeights1 = np.concatenate([npWeights1,npNewHPVec], axis=0)
			newBias1 = np.concatenate([npBias1,npNewHPbias])
			print newBias1.shape

			updateWeights1Op = tf.assign(weights1[currentHiddenDim], newWeights1.T)
			updateWeights2Op = tf.assign(weights2[currentHiddenDim], npNewSelectionWeight.T)
			updateBias1Op = tf.assign(bias1[currentHiddenDim], newBias1)
			updateBias2Op = tf.assign(bias2[currentHiddenDim], npNewSelectionBias)

			sess.run([updateWeights1Op,updateWeights2Op,updateBias1Op,updateBias2Op])
		else:
			print("No hyperplane found")