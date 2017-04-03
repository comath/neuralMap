import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import tensorflow as tf
from PIL import Image
from utils import tile_raster_images

from ipCalculatorWrap import ipCalculator


class nnLayer:
	def __init__(self,inDim,outDim):
		seed = 197098 				#Chosen Randomly
		with tf.name_scope("nnLayer") as scope:
			self.weights = tf.Variable(tf.random_uniform([inDim,outDim], -0.005, 0.005),name="weights")
			self.bias = tf.Variable(tf.zeros([outDim]),name='bias')
	def variables(self):
		return [self.weights, self.bias]
	def eval(self,input):
		return tf.nn.sigmoid(tf.matmul(input,self.weights) + self.bias, name="sigmoid")

def tfNorm(tensor):
	return tf.reduce_sum(tf.square(tensor), [0])

def errorRate(logits, labels):
	labels = tf.to_float(labels, name='ToFloat')
	return tfNorm(logits-labels)

def loss(logits, labels):
	labels = tf.to_float(labels, name='ToFloat')
	return tf.reduce_mean(tfNorm(logits-labels))

def checkSphere(randomPoints,radius):
	norm = np.linalg.norm(randomPoints, axis=1)
	return 0.5+0.5*np.sign(radius-norm)

def cartProd(interval1,interval2):
	return np.transpose([np.tile(interval1, len(interval2)), np.repeat(interval2, len(interval1))])


def createData(batchSize):
	data = np.random.uniform(low=-1.0, high=1.0, size=[batchSize,2])
	return data, np.reshape(checkSphere(data,0.5),[batchSize,1])

batchSize = 100
inputDim = 2
hiddenDim1 = 3
outDim = 1

resolution = 210
xy = np.mgrid[-1:1.1:0.01, -1:1.1:0.01].T
xy = np.ascontiguousarray(xy, dtype=np.float32)
print xy.shape

layer0 = nnLayer(inputDim,hiddenDim1)
layer1 = nnLayer(hiddenDim1,outDim)

X = tf.placeholder(tf.float32,[batchSize,inputDim], name="input")
Y = tf.placeholder(tf.int32,[batchSize,outDim], name="output")

output = layer1.eval(layer0.eval(X))

loss = loss(output,Y)

optimizer = tf.train.GradientDescentOptimizer(0.05)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

tfXY = tf.placeholder(tf.float32,[resolution,resolution,2], name="rendInput")
renOutput = layer1.eval(layer0.eval(tf.reshape(tfXY, [resolution*resolution, 2])))
renderNN = tf.reshape(renOutput, [resolution,resolution])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1, 10000):
	tr_x, tr_y = createData(batchSize)
	sess.run(train_op, feed_dict={X: tr_x, Y:tr_y})

imageArray = sess.run(renderNN,feed_dict={tfXY : xy})
I8 = (((imageArray - imageArray.min()) / (imageArray.max() - imageArray.min())) * 255.9).astype(np.uint8)
img = Image.fromarray(I8)
img.save("file.png")

A,b = sess.run(layer0.variables())
A = np.ascontiguousarray(A.T, dtype=np.float32)
ipCalc = ipCalculator(A,b,2)

imageArray2 = ipCalc.visualize2d(xy,1)
I8 = (imageArray2 * 255.9).astype(np.uint8)
img = Image.fromarray(I8)
img.save("file2.png")