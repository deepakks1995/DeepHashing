import data as dt
from loadImages import load_data
import network as nt
from keras.models import Model
import tensorflow as tf
import numpy as np
import keras.backend as K
import copy, random
import gc, sys 

epochs = 3
k_bit = 64
S = 1 ##change the value of S on each image change
positive_samples = 8
neg_to_pos_ratio = 2
batch_size = 2
step_per_epoch = 3


def train_on_batch(data_manager, model_vars, prim_model, sec_model, prim_inter_model, sec_inter_model, binary_manager):
	while(data_manager._has_next()):
		batch = data_manager._next_batch(4, 8)
		for itr in batch:
			counter = 0
			positive_images, negative_images, pidx_list, nidx_list = itr[0], itr[1], itr[2], itr[3]
			
			model_vars.S = 1
			for i in xrange(0):
				for j in xrange(1, len(positive_images)):
					counter += 1
					model_vars._change_B(pidx_list[i][0], pidx_list[i][1], pidx_list[j][0], pidx_list[j][1])
					prim_model.fit(positive_images[i], temp_label, epochs=1, verbose=0)
					sec_model.fit(positive_images[j], temp_label, epochs=1, verbose=0)
					xvar = (prim_inter_model.predict(positive_images[i])).reshape(model_vars.kbit)
					yvar = (sec_inter_model.predict(positive_images[j])).reshape(model_vars.kbit)
					model_vars.set_column_U(pidx_list[i][0]*positive_samples + pidx_list[i][1], xvar)
					model_vars.set_column_V(pidx_list[j][0]*positive_samples + pidx_list[j][1], yvar)
					model_vars._calculate_binary([prim_model, sec_model], [pidx_list[i][0]*positive_samples + pidx_list[i][1], pidx_list[j][0]*positive_samples + pidx_list[j][1]])
					del xvar, yvar

			model_vars.S = -1
			for j in xrange(len(negative_images)):
				i = 0
				# j = random.randint(0, len(negative_images)-1)
				model_vars._change_B(pidx_list[i][0], pidx_list[i][1], nidx_list[j][0], nidx_list[j][1])
				prim_model.fit(positive_images[i], temp_label, epochs=1, verbose=0)
				sec_model.fit(negative_images[j], temp_label, epochs=1, verbose=0)
				xvar = (prim_inter_model.predict(positive_images[i])).reshape(model_vars.kbit)
				yvar = (sec_inter_model.predict(negative_images[j])).reshape(model_vars.kbit)
				model_vars.set_column_U(pidx_list[i][0]*positive_samples + pidx_list[i][1], xvar)
				model_vars.set_column_V(nidx_list[j][0]*positive_samples + nidx_list[j][1], yvar)
				model_vars._calculate_binary([prim_model, sec_model], [pidx_list[i][0]*positive_samples + pidx_list[i][1], nidx_list[j][0]*positive_samples + nidx_list[j][1]])
				del xvar, yvar

		binary_manager.process_batch(model_vars, batch)
		print "Batch processed completely......."
		del batch
	return 0

if __name__ == '__main__':
	data_manager = dt.DataManager(path='data/FVC2002', similar_samples=positive_samples, batch_size=batch_size, steps_per_epoch=step_per_epoch)

	model_vars = nt.Variables(k_bit, data_manager.total_subjects, positive_samples)
	prim_model = nt.Network(model_vars, 0).generate_model()
	sec_model = nt.Network(model_vars, 1).generate_model()
	prim_inter_model = Model(inputs=prim_model.input, outputs=prim_model.get_layer('Dense11').output)
	sec_inter_model = Model(inputs=sec_model.input, outputs=sec_model.get_layer('Dense11').output)
	binary_manager = nt.BinaryManager()

	temp_label = np.zeros(1)			
	for epoch in xrange(epochs):
		print "Epoch ", epoch
		train_on_batch(data_manager, model_vars, prim_model, sec_model, prim_inter_model, sec_inter_model, binary_manager)
		gc.collect()
		sys.stdout.flush()
		print "Saving Weights.....", epoch
		prim_model.save_weights("models/" + "pri_epochs: " + str(epoch) + ".h5")
		sec_model.save_weights("models/" + "sec_epochs: " + str(epoch) + ".h5")
		data_manager._on_epoch_end()
	print "Training Finished..."