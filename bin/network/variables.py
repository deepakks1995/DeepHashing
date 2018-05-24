import copy

import numpy as np
import tensorflow as tf
from keras.layers import Input


class Variables(object):
    """docstring for Variables"""
    def __init__(self, kbit, length, samples):
        super(Variables, self).__init__()
        self.kbit = kbit
        self._length = length
        self._samples = samples
        self.total_images = self._length*self._samples
        self.B, self.U, self.V, self.W = None, None, None, None
        self.similarity_matrix, self.pri_tensor, self.sec_tensor = None, None, None
        self.gamma, self.tau, self.neta, self.S = 1, 1, 1, 1
        self.__init_vars()

    def __init_vars(self):
        self.B = np.full((self.kbit, self.total_images), 0, dtype=float)
        self.U = np.full((self.kbit, self.total_images), 0, dtype=float)
        self.V = np.full((self.kbit, self.total_images), 0, dtype=float)
        self.W = np.full((self.kbit, self.total_images), 0, dtype=float)
        self.similarity_matrix = np.full((self.total_images, self.total_images), -1, dtype=int)

        for i in range(self._length):
            for j in range(self._samples):
                self.similarity_matrix[i][i*self._samples + j] = 1

        self.pri_tensor = Input(tensor=tf.zeros(shape=self.kbit, dtype=tf.float32))
        self.sec_tensor = Input(tensor=tf.zeros(shape=self.kbit, dtype=tf.float32))

    def _change_binary(self, prim_index1=0, prim_index2=0, sec_index1=0, sec_index2=0):
        del self.pri_tensor, self.sec_tensor
        self.pri_tensor = Input(tensor=tf.Variable(self.B[:, (prim_index1*self._samples) + prim_index2], dtype=tf.float32) )
        self.sec_tensor = Input(tensor=tf.Variable(self.B[:, (sec_index1*self._samples) + sec_index2] ), dtype=tf.float32)

    def set_column_u(self, index, list):
        for itr in range(self.kbit):
            self.U[itr][index] = copy.deepcopy(list[itr] )
        del list

    def set_column_v(self, index, list):
        for itr in range(self.kbit):
            self.V[itr][index] = copy.deepcopy(list[itr] )
        del list

    def _calculate_binary_hash(self, model, batch):

        for index in batch:
            Q = -2*self.kbit*(np.dot(self.similarity_matrix.transpose(), self.U.transpose()) + np.dot(self.similarity_matrix.transpose(), self.V.transpose())) -2*self.gamma*(self.U.transpose() + self.V.transpose())

            Q_star_c = Q[index].reshape(self.kbit, 1)
            U_star_c = self.U[:, index].reshape(self.kbit, 1)
            V_star_c = self.V[:, index].reshape(self.kbit, 1)

            U_temp = np.concatenate( (self.U[:, 0:index], self.U[:, index+1: self.total_images]), axis=1)
            V_temp = np.concatenate( (self.V[:, 0:index], self.V[:, index+1: self.total_images]), axis=1)
            B_temp = np.concatenate( (self.B[:, 0:index], self.V[:, index+1: self.total_images]), axis=1)

            B_star_c = -1*(2*B_temp.dot(U_temp.transpose().dot(U_star_c) + V_temp.transpose().dot(V_star_c)) + Q_star_c ).reshape(self.kbit)

            for itr in range(self.kbit):
                if B_star_c[itr] > 0:
                    self.B[itr][index] = 1
                else:
                    self.B[itr][index] = 0
            del Q_star_c, U_star_c, V_star_c, B_star_c, Q, U_temp, V_temp, B_temp
        return 0