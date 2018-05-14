from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import Dense, Input, merge, UpSampling2D, Conv2D
from keras.optimizers import SGD,Adam,Adadelta
from sklearn.model_selection import train_test_split
from scipy import ndimage
from time import gmtime,strftime
import numpy as np
import keras
import keras.backend as K
import shutil
import random as rng
import tensorflow as tf
import cv2
import os



num_bits = M = k = 16
batch_size = N = 24
rho = 1
x_layer_size = d = 4096
xb_layer_size = x_layer_size+num_bits
alpha = 1./(num_bits*batch_size)
rotation_samples_step = 1

data_root_dir="data/IITK"
num_subjects, num_train_poses, num_test_poses = 100, 6, 2

def load_data():
    training_images_path = []

    for folder_name in os.listdir(data_root_dir)[:1]:
        # print folder_name
        for file in os.listdir(os.path.join(data_root_dir,folder_name)):
            # print file
            training_images_path.append(os.path.join(data_root_dir, folder_name, file))

            # current_image = cv2.imread(os.path.join(data_root_dir, folder_name, file))
            # current_image = current_image.reshape(224,224,3)
            # current_image = current_image.astype('float64') / 255.
            # training_images.append(current_image)
    # training_images_path = np.asarray(training_images_path)
    return training_images_path

training_images_path = load_data()
print len(training_images_path)

def make_batch(training_images_path,iter_count,batch_size):
    full_size = len(training_images_path)
    start_idx = int(full_size/batch_size)*iter_count
    batch = []
    for i in xrange(start_idx,start_idx+batch_size):
        current_image = cv2.imread(training_images_path[i%full_size])
        current_image = current_image.reshape(224,224,3)
        current_image = current_image.astype('float64') / 255.
        batch.append(current_image)
        for j in xrange(rotation_samples_step):
            ccw = rng.choice([-1.,1.])
            rotation_angle = ccw*((rng.random()-1.)*5.)
            rotated_image = ndimage.rotate(current_image, rotation_angle, reshape=False, cval=255)
            batch.append(rotated_image/255.)
    batch = np.asarray(batch)
    return batch

def decoder(input_shape):
    input_layer = Input(shape=input_shape)
    l1 = Conv2D(512,(3,3),activation='relu',padding='same')(input_layer)
    l1 = Conv2D(512,(3,3),activation='relu',padding='same')(l1)
    l1 = Conv2D(512,(3,3),activation='relu',padding='same')(l1)
    l1 = Conv2D(512,(3,3),activation='relu',padding='same')(l1)
    l2 = Conv2D(512,(3,3),activation='relu',padding='same')(l1)
    l2 = UpSampling2D()(l2)
    l2 = Conv2D(512,(3,3),activation='relu',padding='same')(l2)
    l2 = Conv2D(512,(3,3),activation='relu',padding='same')(l2)
    l2 = Conv2D(512,(3,3),activation='relu',padding='same')(l2)
    l3 = Conv2D(256,(3,3),activation='relu',padding='same')(l2)
    l3 = UpSampling2D()(l3)
    l3 = Conv2D(256,(3,3),activation='relu',padding='same')(l3)
    l3 = Conv2D(256,(3,3),activation='relu',padding='same')(l3)
    l3 = Conv2D(256,(3,3),activation='relu',padding='same')(l3)
    l4 = Conv2D(128,(3,3),activation='relu',padding='same')(l3)
    l4 = UpSampling2D()(l4)
    l4 = Conv2D(128,(3,3),activation='relu',padding='same')(l4)
    l5 = Conv2D(64,(3,3),activation='relu',padding='same')(l4)
    l5 = UpSampling2D()(l5)
    l5 = Conv2D(64,(3,3),activation='relu',padding='same')(l5)
    l5 = Conv2D(64,(3,3),activation='relu',padding='same')(l5)
    final = Conv2D(3,(1,1),activation='sigmoid',padding='same')(l5)
    model = Model(input=input_layer, output=final)
    return model

vgg_model = VGG19(weights='imagenet', input_shape=(224, 224, 3))
# vgg_model.load_weights("vgg19_weights.h5")
vgg_output = vgg_model.get_layer('block5_conv4').output
middle_shape = (14, 14, 512)
decoder_model = decoder(middle_shape)
# decoder_model.summary()

autoencoder_model = Model(inputs = vgg_model.input, outputs = decoder_model(vgg_output))
# autoencoder_model.summary()
# exit()

autoencoder_model.compile(loss ='mean_squared_error',optimizer = Adam(0.00005))

for epoch in xrange(100):
    print "Epoch #",epoch
    batch_size = 16
    iteration_total = 1+int(len(training_images_path)/batch_size)
    # print 'iteration_total', iteration_total
    for iter_count in xrange(iteration_total):
        # print epoch,'-',iter_count,':'
        batch = make_batch(training_images_path,iter_count,batch_size)
        autoencoder_model.train_on_batch(batch, batch)
    if epoch % 10 ==0:
        # autoencoder_model.save_weights("models/vgg_trained_ae_iitk"+strftime('%Y-%m-%d::%H:%M:%S',gmtime())+".h5")
        autoencoder_model.save_weights("models/vgg_trained_ae_iitk.h5")
        vgg_model.save_weights("models/vgg_iitk.h5")

# autoencoder_model.fit(x=training_images, y=training_images, batch_size=32, epochs=100, verbose=2, validation_split=0.2, shuffle=True)
autoencoder_model.save_weights("models/vgg_trained_ae_iitk.h5")
vgg_model.save_weights("models/vgg_iitk.h5")


# vgg_model.summary()

# base_x_layer = vgg_model.get_layer('fc2').output
# base_b_layer = Dense(num_bits, activation='sigmoid')(base_x_layer)
# base_xb_layer = keras.layers.concatenate([base_x_layer, base_b_layer])

# base_model = Model(inputs=vgg_model.input,output=base_xb_layer)
# base_model.summary()

# d

# img = cv2.imread("FVC2002/Db1/1_7")
# img = img.reshape(1,224,224,3)
# img = img.astype('float64') / 255.
# print quant_info_model.predict(img)[0][:,-num_bits:]


# # for subject in xrange(num_subjects):
# for subject in xrange(10):
#     print "Subject #",subject+1
#     batch = make_batch(training_images,subject)
#     dummy = np.zeros((batch_size,x_layer_size+num_bits))
#     print "QuantInfoLoss",quant_info_model.train_on_batch([batch],[dummy]*2)
    
#     batch_repeat,batch_tiled = make_batch_for_semantic(batch)
#     dummy = np.zeros((batch_repeat.shape[0],2*(xb_layer_size)))
#     print "SemanticLoss",semantic_model.train_on_batch([batch_repeat,batch_tiled],dummy)
    
#     # batch_bases,batch_tilted = make_batch_for_rotation(batch)
#     # dummy = np.zeros((batch_bases.shape[0],2*(x_layer_size+num_bits)))
#     # print "RotationLoss",rotation_model.train_on_batch([batch_bases,batch_tilted],dummy)


# img = cv2.imread("FVC2002/Db1/1_7")
# img = img.reshape(1,224,224,3)
# img = img.astype('float64') / 255.
# print quant_info_model.predict(img)[0][:,-num_bits:]

