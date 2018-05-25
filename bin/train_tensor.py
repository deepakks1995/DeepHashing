import numpy as np
from keras.models import Model
import sys
import data as dt
import network as nt

epochs = 70
k_bit = 64
positive_samples = 8
neg_to_pos_ratio = 2
batch_size = 25
step_per_epoch = 500


if __name__ == '__main__':
    data_manager = dt.DataManager(path='data/FVC2002', similar_samples=positive_samples, batch_size=batch_size,
                                  steps_per_epoch=step_per_epoch)

    model_vars = nt.Variables(k_bit, data_manager.total_subjects, positive_samples, batch_size)
    vgg_model, siamese_model = nt.Network(model_vars).generate_model()
    vgg_target = [np.full((batch_size, ), 1, dtype=float)]
    siamese_target = [np.full((batch_size, batch_size), 1, dtype=float), np.full((batch_size, batch_size), -1, dtype=float)]
    data_manager.model_vars = model_vars
    binary_manager = nt.BinaryManager()

    vgg_interim_model = Model(inputs=vgg_model.input, outputs=vgg_model.output)
    siamese_interim_model = Model(inputs=siamese_model.input, outputs=siamese_model.output)

    for epoch in range(epochs):
        print ("Epoch ", epoch)
        while data_manager.has_next():
            vgg_batch, siamese_batch, idx_list = data_manager.next_batch()
            var = vgg_model.train_on_batch(vgg_batch, vgg_target)
            var2 = siamese_model.train_on_batch(siamese_batch, siamese_target)
            print (var)
            print (var2)
            xvar = vgg_interim_model.predict_on_batch(vgg_batch)
            yvar = siamese_interim_model.predict_on_batch(siamese_batch)

            model_vars.set_column_u(xvar, idx_list)
            model_vars.set_column_v(yvar[0], idx_list)
            model_vars.set_column_w(yvar[1], idx_list)
            model_vars._calculate_binary_hash()
            binary_manager.process_batch(model_vars, idx_list)
            sys.stdout.flush()
            del vgg_batch, siamese_batch, idx_list, xvar, yvar
        data_manager.on_epoch_end()
        binary_manager.process_dataset(model_vars)
        vgg_model.save_weights("models/" + "VGG_epochs: " + str(epoch) + ".h5")
        siamese_model.save_weights("models/" + "Siamese_epochs: " + str(epoch) + ".h5")