import copy

import numpy as np


class Variables(object):
    """docstring for Variables"""
    def __init__(self, kbit, length, samples, batch_size):
        super(Variables, self).__init__()
        self.kbit = kbit
        self._length = length
        self._samples = samples
        self.batch_size = batch_size
        self.total_images = self._length*self._samples
        self.B, self.U, self.V, self.W = None, None, None, None
        self.similarity_matrix, self.pri_tensor, self.sec_tensor = None, None, None
        self.gamma, self.tau, self.neta, self.S = 100, 10, 10, 1
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

    def set_column_u(self, values, idx_list):

        for it in range(self.batch_size):
            for itr in range(self.kbit):
                self.U[itr][idx_list[it][0]] = copy.deepcopy(values[it][itr])

    def set_column_v(self, values, idx_list):
        for it in range(self.batch_size):
            for itr in range(self.kbit):
                self.V[itr][idx_list[it][1]] = copy.deepcopy(values[it][itr])

    def set_column_w(self, values, idx_list):
        for it in range(self.batch_size):
            for itr in range(self.kbit):
                self.W[itr][idx_list[it][2]] = copy.deepcopy(values[it][itr])

    def _calculate_binary_hash(self):

        for index in range(self.kbit):
            Q = -2*self.kbit*(np.dot(self.similarity_matrix.transpose(), self.U.transpose()) +
                              np.dot(self.similarity_matrix.transpose(), self.V.transpose()) +
                              np.dot(self.similarity_matrix.transpose(), self.W.transpose())) - \
                2*self.gamma*(self.U.transpose() + self.V.transpose() + self.W.transpose())

            Q_star_c = Q[:, index].reshape(self.total_images, 1)
            U_star_c = self.U.transpose()[:, index].reshape(self.total_images, 1)
            V_star_c = self.V.transpose()[:, index].reshape(self.total_images, 1)
            W_star_c = self.W.transpose()[:, index].reshape(self.total_images, 1)

            U_temp = np.concatenate((self.U.transpose()[:, 0:index], self.U.transpose()[:, index+1: self.kbit]), axis=1)
            V_temp = np.concatenate((self.V.transpose()[:, 0:index], self.V.transpose()[:, index+1: self.kbit]), axis=1)
            W_temp = np.concatenate((self.W.transpose()[:, 0:index], self.W.transpose()[:, index+1: self.kbit]), axis=1)
            B_temp = np.concatenate((self.B.transpose()[:, 0:index], self.B.transpose()[:, index+1: self.kbit]), axis=1)

            # print (Q_star_c)
            B_star_c = -1*(2*B_temp.dot(U_temp.transpose().dot(U_star_c) + V_temp.transpose().dot(V_star_c) +
                                        W_temp.transpose().dot(W_star_c)) + Q_star_c).reshape(self.total_images)
            # print (B_star_c)
            for itr in range(self.total_images):
                if B_star_c[itr] > 0:
                    self.B[index][itr] = 1
                else:
                    self.B[index][itr] = 0
            del Q_star_c, U_star_c, V_star_c, B_star_c, Q, U_temp, V_temp, B_temp
        return 0