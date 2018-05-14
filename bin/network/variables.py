from keras.layers import Input
import tensorflow as tf
import numpy as np

class Variables(object):
 	"""docstring for Variables"""
 	def __init__(self, kbit, length, samples):
 		super(Variables, self).__init__()
 		self.kbit = kbit
 		self._length = length
 		self._samples = samples
 		self.total_images = self._length*self._samples
 		self.B, self.U, self.V = None, None, None
 		self.similarity_matrix, self.pri_tensor, self.sec_tensor = None, None, None
 		self.gamma, self.tau, self.neta, self.S = 1, 1, 1, 1
 		self.__init_vars()

 	def __init_vars(self):
		self.B = tf.Variable(np.full((self.kbit, self.total_images), 0, dtype=float), dtype=tf.float32, name="Binary_Codes")
		self.U = tf.Variable(np.full((self.kbit, self.total_images), 0, dtype=float), dtype=tf.float32, name="Primary_Hash")
		self.V = tf.Variable(np.full((self.kbit, self.total_images), 0, dtype=float), dtype=tf.float32, name="Secondary_Hash")
		self.similarity_matrix = np.full((self.total_images, self.total_images), -1, dtype=int)
				
		for i in xrange(self._length):
			for j in xrange(self._samples):
				self.similarity_matrix[i][i*self._samples + j] = 1
		self.similarity_matrix = tf.Variable(self.similarity_matrix, dtype=tf.float32, name="Similarity_Matrix")
		
		self.pri_tensor = Input(tensor=tf.zeros(shape=self.kbit, dtype=tf.float32))
		self.sec_tensor = Input(tensor=tf.zeros(shape=self.kbit, dtype=tf.float32))

 
 	def _change_B(self, prim_index1=0, prim_index2=0, sec_index1=0, sec_index2=0):
 		del self.pri_tensor, self.sec_tensor
 		self.pri_tensor = Input(tensor=self.B[:, (prim_index1*self._samples) + prim_index2] )
		self.sec_tensor = Input(tensor=self.B[:, (sec_index1*self._samples) + sec_index2] )	

	def _calculate_binary(self, model, variable):

		pri_index = variable[0]*self._samples + variable[1]
		sec_index = variable[2]*self._samples + variable[3]
		range_allowed 	= 	[ (pri_index+i)%self.total_images for i in xrange((int)(0.005*self.total_images) )] \
						+ 	[(sec_index+i)%self.total_images for i in xrange((int)(0.005*self.total_images) )]
		
		for index in [pri_index, sec_index]:

			Q = tf.subtract( tf.scalar_mul(-2*self.kbit,	\
				tf.add(tf.matmul(self.similarity_matrix, self.U, transpose_a=True, transpose_b=True),	\
				tf.matmul(self.similarity_matrix, self.V, transpose_a=True, transpose_b=True) ) ),	\
				tf.scalar_mul(-2*self.gamma,(tf.add(tf.transpose(self.U), tf.transpose(self.V)))) )
			

			Q_star_c =	tf.reshape(tf.transpose(Q)[:, (index)], [self.kbit, 1] )
			U_star_c =	tf.reshape(self.U[:, (index)], [self.kbit, 1] )
			V_star_c =	tf.reshape(self.V[:, (index)], [self.kbit, 1] )
			
			self.U = tf.concat( [ self.U[:, 0:index], self.U[:, index+1: self.total_images]] , axis=1)
			self.V = tf.concat( [ self.V[:, 0:index], self.V[:, index+1: self.total_images]] , axis=1)
			self.B = tf.concat( [ self.B[:, 0:index], self.B[:, index+1: self.total_images]] , axis=1)

			B_star_c =	tf.scalar_mul(-1, \
						tf.sign(tf.add(tf.matmul(tf.scalar_mul(2, self.B), \
						tf.add(tf.matmul(self.U, U_star_c, transpose_a=True), tf.matmul(self.V, V_star_c, transpose_a=True)) ) , Q_star_c)) )

			self.U = tf.concat( [ self.U[:, 0:index], tf.concat( [U_star_c, self.U[:, index:self.total_images]], axis=1)], axis=1)
			self.V = tf.concat( [ self.V[:, 0:index], tf.concat( [V_star_c, self.V[:, index:self.total_images]], axis=1)], axis=1)
			self.B = tf.concat( [ self.B[:, 0:index], tf.concat( [B_star_c, self.B[:, index:self.total_images]], axis=1)], axis=1)
			
			del Q_star_c, U_star_c, V_star_c, B_star_c, Q
		del range_allowed, pri_index, sec_index
		return 0	