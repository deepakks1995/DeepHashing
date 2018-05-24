import data as dt
import network as nt
import numpy as np

epochs = 3
k_bit = 64
positive_samples = 8
neg_to_pos_ratio = 2
batch_size = 15
step_per_epoch = 100


if __name__ == '__main__':
    data_manager = dt.DataManager(path='data/FVC2002', similar_samples=positive_samples, batch_size=batch_size,
                                  steps_per_epoch=step_per_epoch)

    model_vars = nt.Variables(k_bit, data_manager.total_subjects, positive_samples)
    vgg_model, siamese_model = nt.Network(model_vars).generate_model()
    vgg_target = [np.full((batch_size, ), 1, dtype=float)]
    siamese_target = [np.full((batch_size, ), 1, dtype=float), np.full((batch_size, ), -1, dtype=float)]
    data_manager.model_vars = model_vars

    for epoch in range(epochs):
        while data_manager.has_next():
            vgg_batch, siamese_batch = data_manager.next_batch()
            # print (batch[0].shape)
            # print (batch[3].shape)
            vgg_model.train_on_batch(vgg_batch, vgg_target)
            siamese_model.train_on_batch(siamese_batch, siamese_target)
            print ("One Ended")
            del vgg_batch, siamese_batch
        data_manager.on_epoch_end()