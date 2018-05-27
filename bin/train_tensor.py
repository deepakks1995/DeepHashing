import numpy as np
from keras.models import Model
import sys
import data as dt
import network as nt

epochs = 70
k_bit = 64
positive_samples = 8
neg_to_pos_ratio = 2
batch_size = 1
step_per_epoch = 2


if __name__ == '__main__':
    data_manager = dt.DataManager(path='data/FVC2002', similar_samples=positive_samples, batch_size=batch_size,
                                  steps_per_epoch=step_per_epoch)

    model_vars = nt.Variables(k_bit, data_manager.total_subjects, positive_samples, batch_size)
    network = nt.Network(model_vars)
    prim_model = network.generate_model()
    sec_model = network.generate_model()
    vgg_target = [np.full((batch_size, ), 1, dtype=float)]
    siamese_target = [np.zeros((batch_size, ))]
    data_manager.model_vars = model_vars
    binary_manager = nt.BinaryManager()

    prim_interim_model = Model(inputs=prim_model.input, outputs=prim_model.output)
    sec_interim_model = Model(inputs=sec_model.input, outputs=sec_model.output)

    for epoch in range(epochs):
        print ("Epoch ", epoch)
        while data_manager.has_next():
            prim_batch, sec_batch, idx_list = data_manager.next_batch()
            for i in range(2):
                var = prim_model.train_on_batch(prim_batch[i], vgg_target)
                var1 = sec_model.train_on_batch(sec_batch[i], siamese_target)
                print ('\n\nTraining', i, var)
                print (var1)

                xvar = prim_interim_model.predict_on_batch(prim_batch[i])
                yvar = sec_interim_model.predict_on_batch(sec_batch[i])
                model_vars.set_column_u(xvar, idx_list)
                model_vars.set_column_v(yvar, idx_list)
                model_vars.calculate_binary_hash()
                # binary_manager.process_batch(model_vars, idx_list)
                sys.stdout.flush()
            del prim_batch, sec_batch, idx_list
        data_manager.on_epoch_end()
        binary_manager.process_dataset(model_vars)
        prim_model.save_weights("models/" + "VGG_epochs: " + str(epoch) + ".h5")
        sec_model.save_weights("models/" + "Siamese_epochs: " + str(epoch) + ".h5")