import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mapperWrap import nnMap
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
mnist.test.images, mnist.test.labels

inDim = 28*28
hiddenDim = 15
outDim = 10
batchSize = 300

X = tf.placeholder(tf.float32,[None,inDim],name='Input')
Y = tf.placeholder(tf.int32,[None,outDim], name='Labels')

with tf.name_scope('NeuralNetwork'):
	with tf.name_scope('Layer1'):
		weights1 = tf.Variable(tf.random_uniform([inDim,hiddenDim], -0.005, 0.005),name='Weights')
		bias1 = tf.Variable(tf.random_uniform([hiddenDim], -0.005, 0.005),name='Bias')
		eval1 = tf.nn.relu(tf.matmul(X,weights1) + bias1,name='Evaluation')
	with tf.name_scope('Layer2'):
		weights2 = tf.Variable(tf.random_uniform([hiddenDim,outDim], -0.005, 0.005),name='Weights')
		bias2 = tf.Variable(tf.random_uniform([outDim], -0.005, 0.005),name='Bias')
		eval2 = tf.nn.relu(tf.matmul(eval1,weights2) + bias2,name='Evaluation')

loss = tf.losses.softmax_cross_entropy(Y,eval2)

#Summary Section
with tf.name_scope('total'):
    crossEntropy = tf.reduce_mean(loss)
softmaxLoss = tf.summary.scalar('softmax_loss', crossEntropy)

with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(eval2, 1), tf.argmax(Y, 1))
	with tf.name_scope('correct_prediction'):
		incorrect_prediction = tf.not_equal(tf.argmax(eval2, 1), tf.argmax(Y, 1))
	with tf.name_scope('misclassifications'):
		misclass = tf.cast(incorrect_prediction, tf.int32)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

optimizer = tf.train.GradientDescentOptimizer(0.05)
globalStep = tf.Variable(0, name='globalStep', trainable=False)
trainOp = optimizer.minimize(loss, global_step=globalStep)




sess = tf.Session()
init = tf.global_variables_initializer()
writer = tf.summary.FileWriter('tfLogs/mnist_hid1_%(hiddenDim)d_adaptive1'%{'hiddenDim':hiddenDim}, sess.graph)
sess.run(init)


for i in range(20000):
	tr_x, tr_y = mnist.train.next_batch(batchSize)

	feedDict = {X:tr_x, Y:tr_y}
	summOutput,lossVal,trainVal = sess.run([merged,loss,trainOp], feed_dict = feedDict)

	writer.add_summary(summOutput, i)

	if(i % 1000 == 0 and i>0):
		npWeights1, npBias1, npWeights2,npBias2 = sess.run([weights1,bias1,weights2,bias2])
		errors = sess.run([misclass], feed_dict = {X:trX, Y:trY})
		errors = np.concatenate(errors)

		npWeights1 = np.ascontiguousarray(npWeights1.T, dtype=np.float32)
		npWeights2 = np.ascontiguousarray(npWeights2.T, dtype=np.float32)
		npBias1 = np.ascontiguousarray(npBias1, dtype=np.float32)
		npBias2 = np.ascontiguousarray(npBias2, dtype=np.float32)

		neuralMap = nnMap(npWeights1,npBias1,2)
		indicies = np.arange(trX.shape[0],dtype=np.int32)
		neuralMap.batchAdd(trX,indicies,errors)
		newHPVec, newHPoff, unpackedSigs, labels, selectionIndex = neuralMap.adaptiveStep(trX,npWeights2,npBias2)

