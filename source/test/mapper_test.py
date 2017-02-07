# The full test suite for the ipCalulator. 
# It's mostly graphical for now, there should be some 
# higher dimensional tests using the connectivity of the data.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from mapperWrap import nnMap
from mapperWrap import convertToRGB
import tensorflow as tf


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
	"""Calculates the loss from the logits and the labels.
	Args:
	logits: Logits tensor, float - [batch_size, NUM_CLASSES].
	labels: Labels tensor, int32 - [batch_size].
	Returns:
	loss: Loss tensor of type float.
	"""
	labels = tf.to_float(labels, name='ToFloat')
	return ms(logits-labels)

def checkSphere(randomPoints):
	norm = np.linalg.norm(randomPoints, axis=1)
	return np.sign(norm-3)


inputDim = 10
hiddenDim1 = 10
outDim = 1
numData = 100

idMat = np.identity(inputDim,dtype=np.float32)
originVec = np.zeros([inputDim],dtype=np.float32)
mean = originVec
cov = idMat



layer0 = nnLayer(inputDim,hiddenDim1)
layer1 = nnLayer(hiddenDim1,outDim)


X = tf.placeholder(tf.float32,[numData,inputDim], name="input")
Y = tf.placeholder(tf.int32,[numData,outDim], name="output")

output = layer1.eval(layer0.eval(X))

loss = loss(output,Y)


optimizer = tf.train.GradientDescentOptimizer(0.05)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

weights0,bias0,weights1,bias1 = sess.run([layer0.weights, layer0.bias,layer1.weights,layer1.bias])
map1 = nnMap(weights0,bias0,weights1,bias1,2,0.5)

data = np.random.multivariate_normal(mean,cov,numData).astype(dtype=np.float32)
labels = checkSphere(data)

map1.batchAdd(data,sess.run(errorRate(output,Y),feed_dict={X: data, Y:labels}),1)

ipSignature,regSignature,avgPoint,avgErrorPoint,thisLoc.numPoints,thisLoc.numErrorPoints = map1.location(0)
print ipSignature



		

	
