import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy import io
import tensorflow as tf

from dataFeed import filtered_mnist



inputDim 	= 28*28
outDim 		= 10                         
batchSize 	= 300
numTrainingSteps 	= 20000


nnVars = {}
output = []
lossObj = []
train_op = []

data = filtered_mnist()
images, labels = data.setupTFDataConstants(batchSize)
optimizer = tf.train.GradientDescentOptimizer(0.05)
global_step = tf.Variable(0, name='global_step', trainable=False)

for hiddenDim1 in range(10,101,5):
	with tf.name_scope("hidden_%(hid)d"%{'hid':hiddenDim1}):
		with tf.name_scope("Layer0"):
			weights0 = tf.Variable(tf.random_normal([inputDim,hiddenDim1]),name="weights_0")
			nnVars["matrix%(hid)04dlayer0"% {'hid': hiddenDim1}] = weights0

			bias0 = tf.Variable(tf.random_normal([hiddenDim1]),name='bias_0')
			nnVars["offset%(hid)04dlayer0"% {'hid': hiddenDim1}] = bias0
			evalLayer0 = tf.matmul(images,weights0) + bias0
			outLayer0 = 0.09*tf.nn.relu(evalLayer0) + 0.01*evalLayer0

		with tf.name_scope("Layer1"):
			weights1 = tf.Variable(tf.random_normal([hiddenDim1,outDim]),name="weights_1")
			nnVars["matrix%(hid)04dlayer1"% {'hid': hiddenDim1}] = weights1

			bias1 = tf.Variable(tf.random_normal([outDim]),name='bias_1')
			nnVars["offset%(hid)04dlayer1"% {'hid': hiddenDim1}] = bias1

			evalLayer1 = tf.matmul(outLayer0,weights1) + bias1
			outLayer1 = 0.09*tf.nn.relu(evalLayer1) + 0.01*evalLayer1
			output.append(outLayer1)
		localCrossEntropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outLayer1)
		localLoss = tf.reduce_mean(localCrossEntropy, name='xentropy_mean')
		lossObj.append(localLoss)
		train_op.append(optimizer.minimize(localLoss, global_step=global_step))

		with tf.name_scope('accuracy_%(hid)d'%{'hid':hiddenDim1}):
		  with tf.name_scope('correct_prediction'):
		    correct_prediction = tf.equal(tf.argmax(outLayer1, 1), tf.argmax(labels, 1))
		  with tf.name_scope('accuracy'):
		    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy_%(hid)d'%{'hid':hiddenDim1}, accuracy)
		tf.summary.scalar('loss_%(hid)d'%{'hid':hiddenDim1}, localLoss)

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
		trainVal,summary = sess.run([train_op,merged])
		writer.add_summary(summary, step)
		step +=1
		if(step > numTrainingSteps):
			coord.request_stop()
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()


io.savemat("mnist2Layer4leakyRELU.mat", sess.run(nnVars))
