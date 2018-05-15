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
		for index in xrange(2):
			#Training on positive samples
			for i in xrange(1, positive_samples):
				model_vars.S = 1
				model_vars._change_B(index, 0, index, i)
				prim_model.fit(data[index][0], temp_label, epochs=1, verbose=0)
				sec_model.fit(data[index][i], temp_label, epochs=1, verbose=0)	
	# 			xvar = tf.transpose(tf.Variable(prim_inter_model.predict(data[index][0]), dtype=tf.float32) )
	# 			yvar = tf.transpose(tf.Variable(sec_inter_model.predict(data[index][i]), dtype=tf.float32) )
	# 			model_vars.U = tf.concat([ model_vars.U[:,0:(index*positive_samples + 0)], tf.concat([xvar, model_vars.U[:,(index*positive_samples+1):len(data)*positive_samples] ], axis=1), ], axis=1)
	# 			model_vars.V = tf.concat([ model_vars.V[:,0:(index*positive_samples + i)], tf.concat([yvar, model_vars.V[:,(index*positive_samples+i+1):len(data)*positive_samples] ], axis=1), ], axis=1)
	# 			model_vars._calculate_binary([prim_model, sec_model], [index, 0, index, i])
	# 			del xvar
	# 			del yvar
	# 		print "Training on positive samples completed... ", index+1
	# 		#Training on negative samples
	# 		for i in xrange(1, positive_samples*neg_to_pos_ratio):
	# 			model_vars.S = -1
	# 			model_vars._change_B(index, 0, (index+i)%len(data), 0)
	# 			prim_model.fit(data[index][0], temp_label, epochs=1, verbose=0)
	# 			sec_model.fit(data[(index+i)%len(data)][0], temp_label, epochs=1, verbose=0)
	# 			xvar = tf.transpose(tf.Variable(prim_inter_model.predict(data[index][0]), dtype=tf.float32) )
	# 			yvar = tf.transpose(tf.Variable(sec_inter_model.predict(data[(index+i)%len(data)][0]), dtype=tf.float32) )
	# 			model_vars.U = tf.concat([ model_vars.U[:,0:(index*positive_samples + 0)], tf.concat([xvar, model_vars.U[:,(index*positive_samples+1):len(data)*positive_samples] ], axis=1), ], axis=1)
	# 			model_vars.V = tf.concat([ model_vars.V[:,0:( ((index+i)%len(data))*positive_samples)], tf.concat([yvar, model_vars.V[:,(((index+i)%len(data))*positive_samples+1):len(data)*positive_samples] ], axis=1), ], axis=1)
	# 			model_vars._calculate_binary([prim_model, sec_model], [index, 0, ((index+i)%len(data)), 0])
	# 			del xvar
	# 			del yvar
	# 		print "Training on negative samples completed... ", index+1
	# 		binary_manager._add_items(K.eval(model_vars.B[:, index*positive_samples]) )
	# 		gc.collect()
	# 		sys.stdout.flush()
	# print "Training Finished..."