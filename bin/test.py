import numpy as np
from keras.models import Model
from keras.preprocessing import image
import sys
import data as dt
import network as nt

epochs = 100
k_bit = 64
positive_samples = 80
neg_to_pos_ratio = 2
batch_size = 1
step_per_epoch = 500
comparing_ratio = 1


def calculate_hamming(xvar, yvar):
    result = 0
    for itr in range(k_bit):
        result += abs(xvar[itr] - yvar[itr])
    return result


def get_binary(xvar):
    for itr in range(k_bit):
        # xvar[0][itr] = 0.5*(xvar[0][itr] + yvar[0][itr])
        xvar[0][itr] = 1 if xvar[0][itr] >= 0 else 0
    return xvar[0]


if __name__ == '__main__':
    data_manager = dt.DataManager(path='data/FVC2002', similar_samples=positive_samples, batch_size=batch_size,
                                  steps_per_epoch=step_per_epoch)

    model_vars = nt.Variables(k_bit, data_manager.total_subjects, positive_samples, batch_size)
    network = nt.Network(model_vars)
    prim_model = network.generate_model()
    sec_model = network.generate_model()
    data_manager.model_vars = model_vars
    binary_manager = nt.BinaryManager()
    prim_model.load_weights('saved_weights/VGG_epochs: 0_max_64.h5')
    sec_model.load_weights('saved_weights/Siamese_epochs: 0_max_64.h5')

    anchor_image = image.load_img('data/FVC2002/Db1/1_1.png')
    anchor_image = image.img_to_array(anchor_image)
    anchor_image = np.expand_dims(anchor_image, axis=0)
    positive_image = image.load_img('data/FVC2002/Db1/1_75.png')
    positive_image = image.img_to_array(positive_image)
    positive_image = np.expand_dims(positive_image, axis=0)
    negative_image = image.load_img('data/FVC2002/Db1/75_1.png')
    negative_image = image.img_to_array(negative_image)
    negative_image = np.expand_dims(negative_image, axis=0)

    xvar = prim_model.predict(anchor_image)
    # xvar1 = sec_model.predict(anchor_image)
    binary1 = get_binary(xvar)
    yvar = prim_model.predict(negative_image)
    # yvar1 = sec_model.predict(negative_image)
    binary2 = get_binary(yvar)

    binary_manager._add_items(binary1)
    binary_manager._add_items(binary2)
    print(calculate_hamming(binary1, binary2))