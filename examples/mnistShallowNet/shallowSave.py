import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
mnist.test.images, mnist.test.labels



class nnLayer:
	def __init__(self,inDim,outDim):
		seed = 197098 				#Chosen Randomly
		with tf.name_scope("nnLayer") as scope:
			self.weights = tf.Variable(tf.random_uniform([inDim,outDim], -0.005, 0.005),name="weights")
			self.bias = tf.Variable(tf.zeros([outDim]),name='bias')
	def eval(self,input):
		return tf.nn.sigmoid(tf.matmul(input,self.weights) + self.bias, name="sigmoid")

def tfNorm(tensor):
	return tf.reduce_sum(tf.square(tensor), [0])

def ms(tensor):
	return tf.reduce_mean(tfNorm(tensor))

def errorRate(logits, labels):
	labels = tf.to_float(labels, name='ToFloat')
	return tfNorm(logits-labels)

def loss(logits, labels):
	labels = tf.to_float(labels, name='ToFloat')
	return ms(logits-labels)


num_gpus = 2

inputDim = 28*28
hiddenDim1 = 100
outDim = 10
batchSize = 100


layer0 = []
layer1 = []
output = []
errorRate = []
train_op = []

config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)

X = []
Y = []

matricies = {}

for j in range(1000):
	gpu = j % num_gpus
	with tf.device('/gpu:%d' % gpu):
		X.append(tf.placeholder(tf.float32,[batchSize,inputDim], name="input"))
		Y.append(tf.placeholder(tf.int32,[batchSize,outDim], name="output"))
		layer0.append(nnLayer(inputDim,hiddenDim1))
		layer1.append(nnLayer(hiddenDim1,outDim))
		output.append(layer1[gpu].eval(layer0[gpu].eval(X[gpu])))
		lossObj = loss(output,Y)
		optimizer = tf.train.GradientDescentOptimizer(0.05)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op.append(optimizer.minimize(lossObj, global_step=global_step))

	if( j % num_gpus == num_gpus - 1 and j > 0):
		
		init = tf.global_variables_initializer()
		sess.run(init)
		for k in range(1, 10000):
			tr_x, tr_y  = mnist.train.next_batch(batchSize)
			feed = {}
			for x in X:
				feed[x] = tr_x
			for y in Y:
				feed[y] = tr_y
			sess.run(train_op, feed_dict = feed)

		for i,layer in enumerate(layer0):
			matricies["matrix0batch%(batch)04d"% {'batch': i + j - num_gpus + 1}] = sess.run(layer.weights)
			matricies["offset0batch%(batch)04d"% {'batch': i + j - num_gpus + 1}] = sess.run(layer.bias)
		for i,layer in enumerate(layer1):
			matricies["matrix1batch%(batch)04d"% {'batch': i + j - num_gpus + 1}] = sess.run(layer.weights)
			matricies["offset1batch%(batch)04d"% {'batch': i + j - num_gpus + 1}] = sess.run(layer.bias)
			
		X = []
		Y = []
		layer0 = []
		layer1 = []
		output = []
		errorRate = []
		train_op = []
		tf.reset_default_graph()
		config = tf.ConfigProto(allow_soft_placement = True)
		sess = tf.Session(config = config)

io.savemat("mnist%(visibleDim)dx%(hiddenDim)03dx010.mat" % {'visibleDim': inputDim, 'hiddenDim': hiddenDim1},
													matricies)


		

