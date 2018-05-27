import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Activation, GlobalAveragePooling2D, Dropout
from keras.layers.core import Dense, Lambda
from keras.models import Model

from .squeezenet import SqueezeNet


class Network(object):
    """docstring for Network"""

    def __init__(self, variables, flag=0):
        super(Network, self).__init__()
        self._variables = variables
        self.flag = flag

    '''
        Method to implement Primary and Secondary Network
        :param input_shape: input shape of image
        :return: two model primary and secondary
        '''

    def generate_model(self, input_shape=(224, 224, 3)):
        # VGG Model
        prim_binary_i = Input((self._variables.kbit,), name="Binary_I")
        prim_binary_j = Input((self._variables.kbit,), name="Binary_J")
        hash_j = Input((self._variables.kbit,), name="Hash_J")
        S = Input((self._variables.batch_size,), name="Similarity_Matrix")

        x = SqueezeNet(weights='imagenet', include_top=False, input_shape=input_shape)
        layer = Dropout(0.5, name='drop9')(x.layers[-1].output)
        layer = Conv2D(400, (1, 1), padding='valid', name='conv10')(layer)
        layer = Activation('relu', name='relu_conv10')(layer)
        layer = GlobalAveragePooling2D()(layer)
        layer = Dense(4096, activation='relu', name='fc1')(layer)
        layer = Dense(4096, activation='relu', name='fc2')(layer)
        last_layer = Dense(self._variables.kbit, activation='tanh', name='Dense11')(layer)

        model = Model(inputs=[x.input, prim_binary_i, prim_binary_j, hash_j, S],
                      outputs=[last_layer])
        model.compile(optimizer="adam", loss=self.prim_loss(prim_binary_i, prim_binary_j, hash_j, S))

        for layer in model.layers:
            layer.trainable = False
        model.layers[-1].trainable = True
        model.layers[-2].trainable = True
        model.layers[-3].trainable = True
        model.get_layer('conv10').trainable = True

        return model

    '''
        Primary Network loss function
        :param vgg_binary_i: Binary code for image i
        :param vgg_binary_j: Binary code for image j
        :param vgg_binary_k: Binary code for image k
        :param hash_j: Hash code generated for image j
        :param hash_k: Hash code generated for image k
        :return: 
        '''

    def prim_loss(self, vgg_binary_i, vgg_binary_j, hash_j, S):
        def loss(y_true, y_predict):
            loss1 = (tf.reduce_sum(tf.abs((tf.matmul(K.tanh(tf.cast(y_predict, tf.float64)),
                                                        tf.cast(vgg_binary_j, tf.float64), transpose_b=True) -
                                              tf.scalar_mul(self._variables.kbit, tf.cast(S, tf.float64))))))

            loss2 = self._variables.neta * tf.sqrt(tf.reduce_sum(tf.square(K.tanh(tf.cast(y_predict, tf.float64)))))
            loss3 = self._variables.gamma * tf.sqrt(tf.reduce_sum(tf.square
                                                                  (tf.cast(tf.transpose(y_predict), tf.float64) -
                                                                   tf.cast(tf.transpose(vgg_binary_i), tf.float64))))

            thetha_ij = tf.scalar_mul(0.5, tf.matmul(K.tanh(tf.cast(y_predict, tf.float64)),
                                                     tf.cast(hash_j, tf.float64), transpose_b=True))
            term = tf.log(tf.constant(1.0, dtype=tf.float64) + tf.exp(thetha_ij))
            loss4 = tf.reduce_sum(tf.matmul(tf.cast(S, tf.float64), thetha_ij) - term)
            return tf.cast((loss1 + loss2 + loss3 - loss4), tf.float32)

        return loss
