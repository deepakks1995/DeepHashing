import numpy as np
from keras.models import Model
import sys
import data as dt
import network as nt

epochs = 100
k_bit = 64
positive_samples = 80
neg_to_pos_ratio = 2
batch_size = 1
step_per_epoch = 500
comparing_ratio = 5


def calculate_hamming(xvar, yvar):
    result = 0
    for itr in range(k_bit):
        result += abs(xvar[itr] - yvar[itr])
    return result


def get_binary(xvar, yvar):
    for itr in range(k_bit):
        xvar[0][itr] = 0.5*(xvar[0][itr] + yvar[0][itr])
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

    prim_model.load_weights('saved_weights/VGG_epochs: 0_max_64.h5')
    sec_model.load_weights('saved_weights/Siamese_epochs: 0_max_64.h5')

    for i in range(comparing_ratio):
        prim_batch, sec_batch = data_manager.test_batch([50,8,75])
        for itr in range(2):
            xvar1 = prim_model.predict_on_batch(prim_batch[itr])
            xvar2 = sec_model.predict_on_batch(prim_batch[itr])
            binary1 = get_binary(xvar1, xvar2)
            yvar1 = prim_model.predict_on_batch(sec_batch[itr])
            yvar2 = sec_model.predict_on_batch(sec_batch[itr])
            binary2 = get_binary(yvar1, yvar2)
            result = calculate_hamming(binary1, binary2)
            string = ['Positive', 'Negative']
            print (string[itr], result)
