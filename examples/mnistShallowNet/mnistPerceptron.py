import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy import io
import tensorflow as tf

from dataFeed import filtered_mnist


def nnLayer(self,inDim,outDim,input):
	with tf.name_scope("nnLayer") as scope:
		
		return tf.nn.sigmoid(tf.matmul(input,self.weights) + self.bias, name="sigmoid")

inputDim 	= 28*28
hiddenDim1 	= 60
outDim 		= 10                         
batchSize 	= 300
numEpochs 	= 10000


layer0 = {}
layer1 = {}
output = []
lossObj = []
train_op = []

data = filtered_mnist()
images, labels = data.setupTFDataConstants(numEpochs,batchSize)
optimizer = tf.train.GradientDescentOptimizer(0.01)

for hiddenDim1 in range(20,101,20):
	with tf.name_scope("hidden_%(hid)d"%{'hid':hiddenDim1}):
		with tf.name_scope("Layer0"):
			weights0 = tf.Variable(tf.random_uniform([inputDim,hiddenDim1], -0.005, 0.005),name="weights_0")
			layer0["matrix%(hid)04d"% {'hid': hiddenDim1}] = weights0

			bias0 = tf.Variable(tf.zeros([hiddenDim1]),name='bias_0')
			layer0["offset%(hid)04d"% {'hid': hiddenDim1}] = bias0

			outLayer0 = tf.nn.sigmoid(tf.matmul(images,weights0) + bias0, name="sigmoid")
		with tf.name_scope("Layer1"):
			weights1 = tf.Variable(tf.random_uniform([hiddenDim1,outDim], -0.005, 0.005),name="weights_1")
			layer1["matrix%(hid)04d"% {'hid': hiddenDim1}] = weights1

			bias1 = tf.Variable(tf.zeros([outDim]),name='bias_1')
			layer1["offset%(hid)04d"% {'hid': hiddenDim1}] = bias1

			outLayer1 = tf.nn.sigmoid(tf.matmul(outLayer0,weights1) + bias1, name="sigmoid")
			output.append(outLayer1)
		localLoss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outLayer1)
		with tf.name_scope('accuracy_%(hid)d'%{'hid':hiddenDim1}):
		  with tf.name_scope('correct_prediction'):
		    correct_prediction = tf.equal(tf.argmax(outLayer1, 1), tf.argmax(labels, 1))
		  with tf.name_scope('accuracy'):
		    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy_%(hid)d'%{'hid':hiddenDim1}, accuracy)

merged = tf.summary.merge_all()

config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)
writer = tf.summary.FileWriter('tfLogs', sess.graph)
init_op = tf.group(tf.global_variables_initializer(),
					tf.local_variables_initializer())
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Training
try:
	step = 0
	while not coord.should_stop():
		lossVal,trainVal,summary = sess.run([lossObj,train_op,merged])
		writer.add_summary(summary, step)
		step +=1
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()


layer0 = sess.run(layer0)
layer1 = sess.run(layer1)

io.savemat("mnistLayer0.mat", layer0)
io.savemat("mniatLayer1.mat", layer1)