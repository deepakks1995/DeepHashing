import keras.backend as K
import tensorflow as tf
import numpy as np
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
        vgg_binary_i = Input((self._variables.kbit, ), name="VGG_Binary_I")
        vgg_binary_j = Input((self._variables.kbit, ), name="VGG_Binary_J")
        vgg_binary_k = Input((self._variables.kbit,), name="VGG_Binary_K")
        hash_j = Input((self._variables.kbit,), name="VGG_Hash_J")
        hash_k = Input((self._variables.kbit,), name="VGG_Hash_K")
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
        siamese_binary = Input((self._variables.kbit, ), name="Siamese_Binary")

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
        S_1 = tf.Variable(np.full((1, self._variables.batch_size), 1, dtype=np.float64), dtype=tf.float64)
        S_2 = tf.Variable(np.full((1, self._variables.batch_size), -1, dtype=np.float64), dtype=tf.float64)

        def loss(y_true, y_predict):
            loss1_1 = tf.norm((K.dot(K.tanh(tf.cast(y_predict, tf.float64)),
                                     K.transpose(tf.cast(vgg_binary_j, tf.float64))) -
                               tf.scalar_mul(self._variables.kbit,
                                             tf.Variable(np.full((self._variables.batch_size,
                                                                  self._variables.batch_size), 1, dtype=np.float64),
                                                         dtype=tf.float64))), axis=1)
            loss1_2 = tf.norm((K.dot(K.tanh(tf.cast(y_predict, tf.float64)),
                                     K.transpose(tf.cast(vgg_binary_k, tf.float64))) - tf.scalar_mul(self._variables.kbit,
                                             tf.Variable(np.full((self._variables.batch_size,
                                                                  self._variables.batch_size), -1, dtype=np.float64),
                                                         dtype=tf.float64))), axis=1)

            loss2 = self._variables.neta * tf.norm((K.tanh(tf.cast(y_predict, tf.float64))), axis=1)
            loss3 = self._variables.gamma * tf.norm((tf.cast(y_predict, tf.float64) - tf.cast(vgg_binary_i, tf.float64)),
                                                    axis=1)

            thetha_ij = tf.scalar_mul(0.5, tf.matmul(S_1, tf.matmul(K.tanh(tf.cast(y_predict, tf.float64)),
                                                                    tf.cast(hash_j, tf.float64), transpose_b=True)))
            thetha_ik = tf.scalar_mul(0.5, tf.matmul(S_2, tf.matmul(K.tanh(tf.cast(y_predict, tf.float64)),
                                                                    tf.cast(hash_k, tf.float64), transpose_b=True)))

            loss4_1 = tf.reshape((thetha_ij - tf.log(tf.constant(1.0, dtype=tf.float64) + tf.exp(thetha_ij))), [-1])
            loss4_2 = tf.reshape((thetha_ik - tf.log(tf.constant(1.0, dtype=tf.float64) + tf.exp(thetha_ik))), [-1])
            loss4 = tf.scalar_mul((-1*self._variables.tau), (tf.add(loss4_1, loss4_2)))

            return tf.cast((loss1_1 + loss1_2 + loss2 + loss3 + loss4), tf.float32)
        return loss

    def siamese_loss(self, siamese_binary):

        def loss(y_true, y_predict):
            loss1_1 = tf.norm(((K.dot(K.tanh(y_predict), K.transpose(siamese_binary))) - (self._variables.kbit*y_true)),
                              axis=1)
            loss2 = self._variables.neta * tf.norm((K.tanh(y_predict)), axis=1)
            return tf.add(loss1_1, loss2)
        return loss