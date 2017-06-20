import tensorflow as tf
import numpy as np

def ms(tensor):
	se = tf.reduce_sum(tf.square(tensor), [0])
	return tf.reduce_mean(se)

class RBM:
	def __init__(self, visibleDim, hiddenDim):
		seed = 197098 				#Chosen Randomly
		self.X = tf.placeholder("float",[None,visibleDim])
		self.Y = tf.placeholder("float",[None,hiddenDim])
		with tf.name_scope("RBM") as scope:
			self.weights = tf.Variable(tf.random_uniform([visibleDim,hiddenDim], -0.005, 0.005),name="weights")
			self.vbias = tf.Variable(tf.zeros([hiddenDim]),name='vis_bias')
			self.hbias = tf.Variable(tf.zeros([visibleDim]),name='hid_bias')

	def placeholders(self):
		return self.X, self.Y

	def getWeightsPointer(self):
		return self.weights
	def getVisibleBiasPointer(self):
		return self.vbias
	def getHiddenBiasPointer(self):
		return self.hbias

	def propUp(self,vis):
		return tf.nn.sigmoid(tf.matmul(vis,self.weights) + self.vbias)

	def propDown(self,hid):
		return tf.nn.sigmoid(tf.transpose(tf.matmul(self.weights,tf.transpose(hid)))  + self.hbias)
 
	def reconstruct(self,datum):
		return self.propDown(self.propUp(datum))

	def mse(self,data):
		return ms(data - self.reconstruct(data))

	def contrastiveDivergenceN(self,n=1,rate=0.01):
		h0 = self.propUp(self.X)
		vn = self.X
		hn = h0
		for i in range(n):
			vn = self.propDown(hn)
			hn = self.propUp(vn)
		w_positive_grad = tf.matmul(tf.transpose(self.X),h0)
		w_negative_grad = tf.matmul(tf.transpose(vn),hn)
		update_w = self.weights.assign_add(rate * (w_positive_grad - w_negative_grad))
		update_hb = self.hbias.assign_add(rate * tf.reduce_mean(self.X - vn, 0))
		update_vb = self.vbias.assign_add(rate * tf.reduce_mean(h0 - hn, 0))
		return [update_w,update_vb,update_hb]