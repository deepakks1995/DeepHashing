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
        vgg_binary_i = Input((self._variables.kbit,), name="VGG_Binary_I")
        vgg_binary_j = Input((self._variables.kbit,), name="VGG_Binary_J")
        vgg_binary_k = Input((self._variables.kbit,), name="VGG_Binary_K")
        hash_j = Input((self._variables.kbit,), name="VGG_Hash_J")
        hash_k = Input((self._variables.kbit,), name="VGG_Hash_K")
        x = SqueezeNet(weights='imagenet', include_top=False, input_shape=input_shape)
        layer = Dropout(0.5, name='drop9')(x.layers[-1].output)
        layer = Conv2D(400, (1, 1), padding='valid', name='conv10')(layer)
        layer = Activation('relu', name='relu_conv10')(layer)
        layer = GlobalAveragePooling2D()(layer)
        layer = Dense(4096, activation='relu', name='fc1')(layer)
        layer = Dense(4096, activation='relu', name='fc2')(layer)
        # x = VGG16(weights='imagenet', include_top=True, input_shape=input_shape)
        # x.load_weights('bin/weights/vgg_trip_fvc.h5')
        # x.layers.pop()
        # x.layers.pop()
        last_layer = Dense(self._variables.kbit, activation='tanh', name='Dense11')(layer)
        model_1 = Model(inputs=[x.input, vgg_binary_i, vgg_binary_j, vgg_binary_k, hash_j, hash_k],
                        outputs=[last_layer])
        model_1.compile(optimizer="adam", loss=self.vgg_loss(vgg_binary_i, vgg_binary_j, vgg_binary_k, hash_j, hash_k))
        for layer in model_1.layers:
            layer.trainable = False
        model_1.layers[-1].trainable = True
        model_1.layers[-2].trainable = True
        model_1.layers[-3].trainable = True
        model_1.get_layer('conv10').trainable = True
        # Siamese Model
        left_input = Input(input_shape, name="Left_Siamese")
        right_input = Input(input_shape, name="Right_Siamese")
        binary_i = Input((self._variables.kbit,), name="Siamese_Binary_i")
        binary_j = Input((self._variables.kbit,), name="Siamese_Binary_j")
        binary_k = Input((self._variables.kbit,), name="Siamese_Binary_k")
        hash_i = Input((self._variables.kbit,), name="Siamese_Hash_i")

        convnet_old = SqueezeNet(weights='imagenet', include_top=False, input_shape=input_shape)
        convnet_layers = Dropout(0.5, name='drop9')(convnet_old.layers[-1].output)
        convnet_layers = Conv2D(400, (1, 1), padding='valid', name='conv10')(convnet_layers)
        convnet_layers = Activation('relu', name='relu_conv10')(convnet_layers)
        convnet_layers = GlobalAveragePooling2D()(convnet_layers)
        # convnet_layers = Flatten(name='flatten')(convnet_layers)
        convnet_layers = Dense(4096, activation='relu', name='fc1')(convnet_layers)
        convnet_layers = Dense(4096, activation='relu', name='fc2')(convnet_layers)
        # convnet_old = VGG16(weights='imagenet', include_top=True, input_shape=input_shape)
        # convnet_old.load_weights('bin/weights/vgg_trip_fvc.h5')
        #
        # convnet_old.layers.pop()
        # # convnet_old.layers.pop()
        dense_layer = Dense(self._variables.kbit, activation='tanh', name='Dense21')(convnet_layers)
        convnet = Model(inputs=[convnet_old.input], outputs=[dense_layer])
        for layer in convnet.layers:
            layer.trainable = False
        convnet.layers[-1].trainable = True
        convnet.layers[-2].trainable = True
        convnet.layers[-3].trainable = True
        convnet.get_layer('conv10').trainable = True

        encoded_l = convnet(left_input)
        lambda_1 = Lambda(self.left_loss, output_shape=(self._variables.batch_size, ),
                          arguments={'binary_i': binary_i, 'binary_j': binary_j,
                                     'binary_k': binary_k, 'hash_i': hash_i})(encoded_l)
        encoded_r = convnet(right_input)
        lambda_2 = Lambda(self.right_loss, output_shape=(self._variables.batch_size, ),
                          arguments={'binary_i': binary_i, 'binary_j': binary_j,
                                     'binary_k': binary_k, 'hash_i': hash_i})(encoded_r)
        model_2 = Model(inputs=[left_input, right_input, binary_i, binary_j, binary_k, hash_i],
                        outputs=[encoded_l, encoded_r, lambda_1, lambda_2])
        model_2.compile(optimizer="adam", loss=self.siamese_dummy)
        return model_1, model_2

    '''
        Loss function for left_input in siamese network
        :param y_predict: last layer output
        :param binary_i: binary code learned for index i
        :param binary_j: binary code learned for index j
        :param binary_k: binary code learned for index k
        :param hash_i: hash code learned for index i in first network
        :return:
        '''
    def left_loss(self, y_predict, binary_i, binary_j, binary_k, hash_i):

        S_1 = tf.Variable(np.full((1, self._variables.batch_size), 1, dtype=np.float64), dtype=tf.float64)

        loss1_1 = tf.norm((K.dot(K.tanh(tf.cast(y_predict, tf.float64)),
                                 K.transpose(tf.cast(binary_i, tf.float64))) -
                           tf.scalar_mul(self._variables.kbit,
                                         tf.Variable(np.full((self._variables.batch_size,
                                                              self._variables.batch_size), 1, dtype=np.float64),
                                                     dtype=tf.float64))), axis=1)

        loss2 = self._variables.neta * tf.norm((K.tanh(tf.cast(y_predict, tf.float64))), axis=1)
        loss3 = self._variables.gamma * tf.norm((tf.cast(y_predict, tf.float64) - tf.cast(binary_j, tf.float64)),
                                                axis=1)
        thetha_ij = tf.scalar_mul(0.5, tf.matmul(S_1, tf.matmul(K.tanh(tf.cast(y_predict, tf.float64)),
                                                                tf.cast(hash_i, tf.float64), transpose_b=True)))
        loss4_1 = tf.reshape((thetha_ij - tf.log(tf.constant(1.0, dtype=tf.float64) + tf.exp(thetha_ij))), [-1])

        loss4 = tf.scalar_mul((-1 * self._variables.tau), loss4_1)

        return tf.cast((loss1_1 + loss2 + loss3 + loss4), tf.float32)

    '''
        Loss function for right_input in siamese network
        :param y_predict: last layer output
        :param binary_i: binary code learned for index i
        :param binary_j: binary code learned for index j
        :param binary_k: binary code learned for index k
        :param hash_i: hash code learned for index i in first network
        :return:
        '''
    def right_loss(self, y_predict, binary_i, binary_j, binary_k, hash_i):
        S_2 = tf.Variable(np.full((1, self._variables.batch_size), -1, dtype=np.float64), dtype=tf.float64)

        loss1_2 = tf.norm((K.dot(K.tanh(tf.cast(y_predict, tf.float64)),
                                 K.transpose(tf.cast(binary_i, tf.float64))) -
                           tf.scalar_mul(self._variables.kbit,
                                         tf.Variable(np.full((self._variables.batch_size,
                                                              self._variables.batch_size), -1, dtype=np.float64),
                                                     dtype=tf.float64))), axis=1)

        loss2 = self._variables.neta * tf.norm((K.tanh(tf.cast(y_predict, tf.float64))), axis=1)
        loss3 = self._variables.gamma * tf.norm((tf.cast(y_predict, tf.float64) - tf.cast(binary_k, tf.float64)),
                                                axis=1)
        thetha_ik = tf.scalar_mul(0.5, tf.matmul(S_2, tf.matmul(K.tanh(tf.cast(y_predict, tf.float64)),
                                                                tf.cast(hash_i, tf.float64), transpose_b=True)))

        loss4_2 = tf.reshape((thetha_ik - tf.log(tf.constant(1.0, dtype=tf.float64) + tf.exp(thetha_ik))), [-1])
        loss4 = tf.scalar_mul((-1 * self._variables.tau), loss4_2)

        return tf.cast((loss1_2 + loss2 + loss3 + loss4), tf.float32)

    '''
        Siamese Dummy loss function    
        :param y_true: Fake target values which are not usable in this case
        :param y_predict: loss calculated from previous lambda layers which should be passed from this function
        :return: y_predict
        '''
    def siamese_dummy(self, y_true, y_predict):
        return y_predict

    '''
        Primary Network loss function
        :param vgg_binary_i: Binary code for image i
        :param vgg_binary_j: Binary code for image j
        :param vgg_binary_k: Binary code for image k
        :param hash_j: Hash code generated for image j
        :param hash_k: Hash code generated for image k
        :return: 
        '''
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
                                     K.transpose(tf.cast(vgg_binary_k, tf.float64))) -
                               tf.scalar_mul(self._variables.kbit,
                                             tf.Variable(np.full((self._variables.batch_size,
                                                                  self._variables.batch_size), -1, dtype=np.float64),
                                                         dtype=tf.float64))), axis=1)

            loss2 = self._variables.neta * tf.norm((K.tanh(tf.cast(y_predict, tf.float64))), axis=1)
            loss3 = self._variables.gamma * tf.norm(
                (tf.cast(y_predict, tf.float64) - tf.cast(vgg_binary_i, tf.float64)),
                axis=1)

            thetha_ij = tf.scalar_mul(0.5, tf.matmul(S_1, tf.matmul(K.tanh(tf.cast(y_predict, tf.float64)),
                                                                    tf.cast(hash_j, tf.float64), transpose_b=True)))
            thetha_ik = tf.scalar_mul(0.5, tf.matmul(S_2, tf.matmul(K.tanh(tf.cast(y_predict, tf.float64)),
                                                                    tf.cast(hash_k, tf.float64), transpose_b=True)))

            loss4_1 = tf.reshape((thetha_ij - tf.log(tf.constant(1.0, dtype=tf.float64) + tf.exp(thetha_ij))), [-1])
            loss4_2 = tf.reshape((thetha_ik - tf.log(tf.constant(1.0, dtype=tf.float64) + tf.exp(thetha_ik))), [-1])
            loss4 = tf.scalar_mul((-1 * self._variables.tau), (tf.add(loss4_1, loss4_2)))

            return tf.cast((loss1_1 + loss1_2 + loss2 + loss3 + loss4), tf.float32)

        return loss

    '''
        Depricated function
        Default keras loss function
        :param siamese_binary:
        :return: loss
        '''
    def siamese_loss(self, siamese_binary):

        def loss(y_true, y_predict):
            loss1_1 = tf.norm(
                ((K.dot(K.tanh(y_predict), K.transpose(siamese_binary))) - (self._variables.kbit * y_true)),
                axis=1)
            loss2 = self._variables.neta * tf.norm((K.tanh(y_predict)), axis=1)
            return tf.add(loss1_1, loss2)

        return loss