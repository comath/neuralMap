from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
mnist.test.images, mnist.test.labels



import numpy as np
import scipy.io
import tensorflow as tf

class filtered_mnist:
	def __init__(self):
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
		self.leaveInData = np.stack(leaveInData,axis=0)
		self.leaveInLabel = np.stack(leaveInLabel,axis=0)
		self.removeData = np.stack(removeData,axis=0)
		self.removeLabel = np.stack(removeLabel,axis=0) 
	def imagesDim(self):
		return 28*28
	def labelsDim(self):
		return 10

	def setupTFBatchers(self,num_epochs,batch_size):
		with tf.name_scope('input'):
			# Input data
			self.images_initializer = tf.placeholder(
				dtype=self.leaveInData.dtype,
				shape=self.leaveInData.shape)
			self.labels_initializer = tf.placeholder(
				dtype=self.leaveInLabel.dtype,
				shape=self.leaveInLabel.shape)
			self.tfimages = tf.Variable(
				self.images_initializer, trainable=False, collections=[])
			
			self.tflabels = tf.Variable(
				self.labels_initializer, trainable=False, collections=[])
			
			image, label = tf.train.slice_input_producer(
				[self.tfimages, self.tflabels], num_epochs=num_epochs)
			images, labels = tf.train.batch(
				[image, label], batch_size=batch_size)
			
			return images, labels

	def initTFBatches(self,session):
		session.run(self.tfimages.initializer,feed_dict={self.images_initializer: self.leaveInData})
		session.run(self.tflabels.initializer,feed_dict={self.labels_initializer: self.leaveInLabel})

	def setupTFDataConstants(self,num_epochs,batch_size):
		with tf.name_scope('input'):
			# Input data
			with tf.device('/cpu:0'):
				self.tfimages = tf.constant(self.leaveInData,dtype=tf.float32)
		  		self.tflabels = tf.constant(self.leaveInLabel,dtype=tf.int32)
	  			
			images, labels = tf.train.shuffle_batch(
				[self.tfimages, self.tflabels],batch_size,capacity=50000,
      				min_after_dequeue=10000,
      				num_threads=3,
      				enqueue_many=True)
		return images, labels
	def getLeaveIn():
		return leaveInData, leaveInLabel
	def getRemoved():
		return removeData, removeLabel