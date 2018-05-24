import keras.backend as K
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model


class Network(object):
    """docstring for Network"""

    def __init__(self, variables, flag=0):
        super(Network, self).__init__()
        self._variables = variables
        self.flag = flag

    def generate_model(self, input_shape=(224, 224, 3)):

        # VGG Model
        vgg_binary_i = Input((self._variables.kbit, ), name="VGG_Binary_I", dtype=tf.float32)
        vgg_binary_j = Input((self._variables.kbit, ), name="VGG_Binary_J", dtype=tf.float32)
        vgg_binary_k = Input((self._variables.kbit,), name="VGG_Binary_K", dtype=tf.float32)
        hash_j = Input((self._variables.kbit,), name="VGG_Hash_J", dtype=tf.float32)
        hash_k = Input((self._variables.kbit,), name="VGG_Hash_K", dtype=tf.float32)
        x = VGG16(weights='imagenet', include_top=True, input_shape=input_shape)
        x.layers.pop()
        x.layers.pop()
        last_layer = Dense(self._variables.kbit, activation='tanh', name='Dense11')(x.layers[-1].output)
        model_1 = Model(inputs=[x.input, vgg_binary_i, vgg_binary_j, vgg_binary_k, hash_j, hash_k],
                        outputs=[last_layer])
        model_1.compile(optimizer="adam", loss=self.vgg_loss(vgg_binary_i, vgg_binary_j, vgg_binary_k, hash_j, hash_k))

        # Siamese Model
        left_input = Input(input_shape, name="Left_Siamese")
        right_input = Input(input_shape, name="Right_Siamese")
        siamese_binary = Input((self._variables.kbit, ), name="Siamese_Binary", dtype=tf.float32)

        convnet_old = VGG16(weights='imagenet', include_top=True, input_shape=input_shape)
        convnet_old.layers.pop()
        convnet_old.layers.pop()
        dense_layer = Dense(self._variables.kbit, activation='tanh', name='Dense21')(convnet_old.layers[-1].output)
        convnet = Model(inputs=[convnet_old.input], outputs=[dense_layer])
        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)
        model_2 = Model(inputs=[left_input, right_input, siamese_binary], outputs=[encoded_l, encoded_r])
        model_2.compile(optimizer="adam", loss=self.siamese_loss(siamese_binary))

        return model_1, model_2

    def vgg_loss(self, vgg_binary_i, vgg_binary_j, vgg_binary_k, hash_j, hash_k):

        def loss(y_true, y_predict):
            loss1_1 = tf.norm((K.dot(K.tanh(y_predict), K.transpose(vgg_binary_j))),
                              ord='fro', axis=(0, 1)) - self._variables.kbit
            loss1_2 = tf.norm((K.dot(K.tanh(y_predict), K.transpose(vgg_binary_k))),
                              ord='fro', axis=(0, 1)) + self._variables.kbit
            loss2 = self._variables.neta * tf.norm((K.tanh(K.transpose(y_predict))), ord='fro', axis=(0, 1))
            loss3 = self._variables.gamma * tf.norm((K.transpose(y_predict) - K.transpose(vgg_binary_i)),
                                                    ord='fro', axis=(0, 1))
            thetha_ij = 0.5*tf.reshape(tf.matmul(tf.reshape(K.tanh(y_predict),
                                                            [1, self._variables.kbit]),
                                                 tf.reshape(K.tanh(K.transpose(hash_j)),
                                                            [self._variables.kbit, 1])), [])
            thetha_ik = 0.5*tf.matmul(K.tanh(y_predict), K.tanh(K.transpose(hash_k)))
            loss4_1 = (thetha_ij - tf.log(1 + tf.exp(thetha_ij)))
            loss4_2 = (-thetha_ik - tf.log(1 + tf.exp(thetha_ik)))
            loss4 = -1*self._variables.tau*(loss4_1 + loss4_2)

            return loss1_1 + loss1_2 + loss2 + loss3 + thetha_ij
        return loss

    def siamese_loss(self, siamese_binary):

        def loss(y_true, y_predict):
            loss1_1 = tf.norm(((K.dot(K.tanh(y_predict), K.transpose(siamese_binary))) - (self._variables.kbit*y_true)),
                              ord='fro', axis=(0, 1))
            loss2 = self._variables.neta * tf.norm((K.tanh(K.transpose(y_predict))), ord='fro', axis=(0, 1))
            # loss3 = NotImplemented
            return loss1_1 + loss2
        return loss

    def dummy_loss(self, binary_1, binary_2, f_hash, g_hash, h_hash):
        def loss(y_true, y_predict):
            loss3_1 = tf.norm(((K.tanh(y_predict) * K.transpose(binary_1)) - self._variables.kbit*y_true),
                              ord='fro', axis=(0, 1))
            loss3_2 = tf.norm(((K.tanh(y_predict) * K.transpose(binary_2)) + self._variables.kbit),
                              ord='fro', axis=(0, 1))
            # loss1 = self._variables.gamma * tf.norm((K.tanh(K.transpose(y_predict) ) - K.transpose(binary)), ord='fro', axis=(0, 1))
            # loss2 = self._variables.neta * tf.norm((K.tanh(y_predict)), ord='fro', axis=(0, 1))
            return loss3_1 + loss3_2
        return loss

    def custom_loss(self, tensor):
        def loss(y_true, y_predict):
            loss1 = self._variables.gamma * tf.norm((K.tanh(y_predict) - tensor), ord='fro', axis=(0, 1))
            loss2 = self._variables.neta * tf.norm((K.tanh(y_predict)), ord='fro', axis=(0, 1))
            loss3 = tf.norm((K.tanh(y_predict) * K.transpose(tensor) - self._variables.kbit * self._variables.S),
                            ord='fro', axis=(0, 1))
            return loss1 + loss2 + loss3

        return loss

    def body1(self, i, result):
        j = tf.constant(0)
        condition2 = lambda j, i, result: tf.less(j, self._variables.total_images)

        def body2(j, i, result):
            # theta_i_j = 0.5*tf.reshape(tf.matmul(tf.reshape(self._variables.U[:, i], [1, self._variables.kbit]) , tf.reshape(self._variables.V[:, j], [self._variables.kbit, 1]) ), [])
            # result_j = self._variables.similarity_matrix[i][j]*theta_i_j - tf.log(1 + tf.exp(theta_i_j))
            # return j+1, i, result + result_j
            return j + 1, i, 0

        j, i, result = tf.while_loop(condition2, body2, loop_vars=[j, i, result])
        return i + 1, result
