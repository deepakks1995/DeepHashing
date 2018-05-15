from loadImages import load_data
import network as nt
from keras.models import Model
import tensorflow as tf
import numpy as np
import keras.backend as K
import copy
import gc, sys 

epochs = 1
k_bit = 64
S = 1 ##change the value of S on each image change
positive_samples = 8
neg_to_pos_ratio = 1


if __name__ == '__main__':
	data = load_data('data/FVC2002', positive_samples)

	model_vars = nt.Variables(k_bit, len(data), positive_samples)
	prim_model = nt.Network(model_vars, 0).generate_model()
	sec_model = nt.Network(model_vars, 1).generate_model()
	prim_inter_model = Model(inputs=prim_model.input, outputs=prim_model.get_layer('Dense11').output)
	sec_inter_model = Model(inputs=sec_model.input, outputs=sec_model.get_layer('Dense11').output)
	binary_manager = nt.BinaryManager()

	temp_label = np.zeros(1)			
	for epoch in xrange(epochs):
		print "Epoch ", epoch
		for index in xrange(len(data)):
			#Training on positive samples
			for i in xrange(positive_samples):
				model_vars.S = 1
				model_vars._change_B(index, 0, index, i)
				prim_model.fit(data[index][0], temp_label, epochs=1, verbose=0)
				sec_model.fit(data[index][i], temp_label, epochs=1, verbose=0)	
				xvar = (prim_inter_model.predict(data[index][0])).reshape(model_vars.kbit)
				yvar = (sec_inter_model.predict(data[index][i])).reshape(model_vars.kbit)
				model_vars.set_column_U(index*positive_samples, xvar)
				model_vars.set_column_V(index*positive_samples+i, yvar)
				model_vars._calculate_binary([prim_model, sec_model], [index, 0, index, i])
				del xvar
				del yvar
			print "Training on positive samples completed... ", index+1
			#Training on negative samples
			for i in xrange(1, positive_samples*neg_to_pos_ratio):
				model_vars.S = -1
				model_vars._change_B(index, 0, (index+i)%len(data), 0)
				prim_model.fit(data[index][0], temp_label, epochs=1, verbose=0)
				sec_model.fit(data[(index+i)%len(data)][0], temp_label, epochs=1, verbose=0)
				xvar = (prim_inter_model.predict(data[index][0])).reshape(model_vars.kbit)
				yvar = (sec_inter_model.predict(data[(index+i)%len(data)][0])).reshape(model_vars.kbit)
				model_vars.set_column_U((index*positive_samples + 0), xvar)
				model_vars.set_column_V(( ((index+i)%len(data))*positive_samples), yvar)
				model_vars._calculate_binary([prim_model, sec_model], [index, 0, ((index+i)%len(data)), 0])
				del xvar
				del yvar
			print "Training on negative samples completed... ", index+1
			binary_manager._add_items((model_vars.B[:, index*positive_samples]) )
			gc.collect()
			sys.stdout.flush()
	print "Training Finished..."