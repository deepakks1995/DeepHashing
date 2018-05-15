from keras.applications.vgg16 import VGG16
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import sys

class Network(object):
 	"""docstring for Network"""
 	def __init__(self, variables, flag=0):
 		super(Network, self).__init__()
 		self._variables = variables
 		self.flag = flag

 	def generate_model(self, image_shape=(224, 224, 3)):
 		x = VGG16(weights='imagenet', include_top=True, input_shape=image_shape)
		x.layers.pop()
		x.layers.pop()
		last_layer = Dense(self._variables.kbit, activation='tanh', name='Dense11')(x.layers[-1].output)

		model = Model(inputs=[x.input, (self._variables.pri_tensor if self.flag==0 else self._variables.sec_tensor)], outputs=[last_layer])
		model.compile(optimizer="adam", loss=self.custom_loss(self._variables.pri_tensor if self.flag==0 else self._variables.sec_tensor))
		return model

	def custom_loss(self, tensor):
		
		def loss(y_true, y_predict):
			i = tf.constant(0)
			result = tf.constant(0, dtype=tf.float32)
			condition1 = lambda i, result : tf.less(i, self._variables.total_images)
			# i, result = tf.while_loop(condition1, self.body1, [i, result])
			loss1 =	self._variables.gamma*tf.norm((K.tanh(y_predict) - tensor), ord='fro' ,axis=(0,1))
			loss2 =	self._variables.neta*tf.norm((K.tanh(y_predict)), ord='fro', axis=(0,1))	
			loss3 =	tf.norm((K.tanh(y_predict)*K.transpose(tensor) - self._variables.kbit*self._variables.S), ord='fro', axis=(0,1))	
			# loss1 =	self._variables.gamma*K.l2_normalize((K.tanh(y_predict) - tensor), axis=(0,1))
			# loss2 =	self._variables.neta*K.l2_normalize((K.tanh(y_predict)), axis=-1)	
			# loss3 =	K.l2_normalize((K.tanh(y_predict)*K.transpose(tensor)) - self._variables.kbit*self._variables.S, axis=-1)
			net_loss = loss1 + loss2 + loss3
			return net_loss
		return loss

	def body1(self, i, result):
		j = tf.constant(0)
		condition2 = lambda j, i, result :tf.less(j, self._variables.total_images)

		def body2(j, i, result):
			# theta_i_j = 0.5*tf.reshape(tf.matmul(tf.reshape(self._variables.U[:, i], [1, self._variables.kbit]) , tf.reshape(self._variables.V[:, j], [self._variables.kbit, 1]) ), [])
			# result_j = self._variables.similarity_matrix[i][j]*theta_i_j - tf.log(1 + tf.exp(theta_i_j))
			# return j+1, i, result + result_j
			return j+1, i, 0

		j, i, result = tf.while_loop(condition2, body2, loop_vars=[j, i, result])
		return i+1, result