# python3 this.py int(.model) Word2Vector.model actions
# no.126
import io
import os
import sys
import csv
import string
import numpy as np
from numpy import random
import pandas as pd
from gensim.models import Word2Vec
from gensim import models
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input,LSTM,Dropout,Dense,Activation, GRU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

import keras.backend.tensorflow_backend as K
import tensorflow as tf

from util import StringTools

class RNN():

	def __init__(self):
		self.label = [];
		self.data = [];

	def readfile(self, stoplist):

		inputfile = sys.argv[1];		

		with io.open(inputfile, 'r', encoding='utf-8') as content:
			cnt = 0;
			for line in content:
				cont = line.lower().split('+++$+++');

				if len(cont) == 2:
					self.label.append(int(cont[0]));
					self.data.append([]);
					# cont_split = cont[1].replace('\n','').split(' ');
					cont_split = cont[1].replace('\n','').replace('.','').replace(',','').split(' ');

					cont_split = StringTools.Manage_Abbreviation(cont_split);

					### Delete the punctuations.
					# for idx, word in enumerate(cont_split):
					# 	cont_split[idx] = word.translate(str.maketrans("","", string.punctuation));

					for word in cont_split:
						if word != ' ' and word != '' and word != '\n' and word != '\t' and word not in stoplist:

							new_word = StringTools.Delete_Duplicate_chars(word);

							self.data[cnt].append(new_word)
					cnt += 1;
		
		# remove common words
		# self.data = [[word for word in line if word not in stoplist] for line in self.data];	#[ [a], [b], [c] .......]

		print('--- Readfile Success ---' );

	def TrainData_Manager(self, vector_size, stoplist, nb_part=1, mode='train', part=0.1, maxsize=40):

		if mode == 'train':
			RNN.readfile(self ,stoplist);				

			x_train = np.zeros((len(self.data), maxsize, vector_size))

			print(self.data[41492]);

			for i,words in enumerate(self.data):
				for j in range(len(words)):
					try: 
						x_train[i][maxsize-len(words)+j] = model[words[j]];
					except Exception as e:
						x_train[i][maxsize-len(words)+j] = np.zeros(vector_size);
						pass;
					# else:
					# 	x_train[i][j] = np.zeros(vector_size);
			return x_train

		elif mode == 'semi':

			inputfile = 'training_nolabel.txt';

			# Delete the Puntuation and Stoplist words.
			words_semi = [];
			with io.open(inputfile, 'r', encoding='utf-8') as content:
				for line in content:
					cont = line.lower().replace('\n','').replace('.','').replace(',','').split(' ');

					cont = StringTools.Manage_Abbreviation(cont);

					for idx, word in enumerate(cont):
						cont[idx] = word.translate(str.maketrans("","", string.punctuation));

					words_temp = [];
					for word in cont:
						if word != ' ' and word != '' and word != '\n' and word != '\t' and word not in stoplist:
							new_word = StringTools.Delete_Duplicate_chars(word);
							words_temp.append(new_word)
					words_semi.append(words_temp);
			# words_semi = [[word for word in line if word not in stoplist] for line in words_semi];

			length = int(len(words_semi)*part);
			words_semi = words_semi[nb_part*length:(nb_part+1)*length-1];

			x_train_semi = np.zeros((len(words_semi), maxsize, vector_size))

			for i,words in enumerate(words_semi):
				for j in range(len(words)):
					try: 
						x_train_semi[i][maxsize-len(words)+j] = model[words[j]];
					except Exception as e:
						x_train_semi[i][maxsize-len(words)+j] = np.zeros(vector_size);
						pass;

			print('--- Nolabel Data to Vector Success ---');
			return x_train_semi

	def get_semi_data(x_semi_all, semi_pred, threshold):
		semi_pred = np.squeeze(semi_pred);
		index = (semi_pred > (1-threshold)) + (semi_pred < threshold);
		semi_pred = np.greater(semi_pred, 0.5).astype(np.int32);
		return x_semi_all[index,:], semi_pred[index];

	def model_structure(time_step,input_size, cell='LSTM'):

		inputs = Input(shape=(time_step, input_size));

		if cell == 'LSTM':
			RNN_cell = LSTM(256,
							return_sequences=False,
							dropout=0.3);
		elif cell == 'GRU':
			RNN_cell = GRU(256,	return_sequences=False,	dropout=0.3);

		RNN_output = RNN_cell(inputs);
						# recurrent_regularizer=regularizers.l2(0.1),kernel_regularizer=regularizers.l2(0.1)
		outputs = Dense(256,
						activation='relu')(RNN_output);

		# outputs = Dropout(0.2)(outputs);
		outputs = Dense(1, activation='sigmoid')(RNN_output);

		model_RNN = Model(inputs=inputs, outputs=outputs);
		adam = Adam();
		model_RNN.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

		print(model_RNN.summary())

		return model_RNN


def _shuffle(x, y):
	randomize = np.arange(len(x));
	np.random.shuffle(randomize);
	return x[randomize], y[randomize];
def validation(x, y, percentage):
	all_data_size = len(x);
	train_data_size = int(all_data_size * percentage);
	x_all, y_all = _shuffle(x,y);

	x_train, y_train = x_all[0:train_data_size], y_all[0:train_data_size];
	x_valid, y_valid = x_all[train_data_size:], y_all[train_data_size:];
	return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':

	def get_session(gpu_fraction):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
	K.set_session(get_session(0.6))

	actions = 'train';
	model = models.Word2Vec.load('Word2Vector_rp4.model');
	time_step = 40;

	stoplist = set('of to a b c d e f g h i j l m n o p q r s t u v w x y'.split());

	vector_size = len(model['love'])

	e = RNN();
	x_train = e.TrainData_Manager(vector_size,stoplist, maxsize=time_step);
	y_train = e.label;
	x_train = np.array(x_train);
	y_train = np.array(y_train);
	np.random.seed(100);
	x_train, y_train, x_valid, y_valid = validation(x_train, y_train, 0.9);

	RNN_model = RNN.model_structure(time_step, vector_size, cell='GRU');	

	if actions == 'train':

		checkpoint = ModelCheckpoint(filepath='Rnnmodel_best.h5', 
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_acc',
                                 mode='max' )
		
		train_history = RNN_model.fit(x=x_train,y=y_train,validation_data=(x_valid,y_valid),batch_size = 256, epochs=20, callbacks=[checkpoint]);

		RNN_model.save('Rnnmodel.h5');

	# elif actions == 'semi':

	# 	save_path = sys.argv[1] + '_semi_best.h5';
	# 	checkpoint = ModelCheckpoint(filepath=save_path, 
 #                                 verbose=1,
 #                                 save_best_only=True,
 #                                 monitor='val_acc',
 #                                 mode='max' )

	# 	# train_history = RNN_model.fit(x=x_train,y=y_train,validation_data=(x_valid,y_valid),batch_size = 256, epochs=50, callbacks=[checkpoint]); ######################
	# 	# print('--- Supervised Training Success ---')



	# 	# print('--- Load Nolabel Data ---');
	# 	# x_semi_all = e.TrainData_Manager(vector_size, stoplist, nb_part=1, part = 0.1, mode='semi');
	# 	# print('--- Begin Training ---');
	# 	# for i in range(20):
	# 	# 	semi_pred = RNN_model.predict(x_semi_all, batch_size=1024, verbose=True);
	# 	# 	x_semi, y_semi = RNN.get_semi_data(x_semi_all, semi_pred, 0.05);
	# 	# 	x = np.concatenate((x_train, x_semi));
	# 	# 	y = np.concatenate((y_train, y_semi));
	# 	# 	history = RNN_model.fit(x=x, y=y, validation_data=(x_valid, y_valid), batch_size=256, epochs=1, callbacks=[checkpoint]);
	# 	# 	if os.path.exists(save_path):
	# 	# 		RNN_model.load_weights(save_path);

	# 	for nb_part in range(10): # Split to 10 part for training.

	# 		print('--- Nolabel Data %d Start ---' %nb_part);
	# 		x_semi_all = e.TrainData_Manager(vector_size, stoplist, nb_part=nb_part, mode='semi');
	# 		print('x_semi_all shape: ', x_semi_all.shape);			

	# 		# Secure using the best model parameters.
	# 		if os.path.exists(save_path):
	# 			RNN_model.load_weights(save_path);

	# 		for i in range(3):	# Every part train 3 times.
	# 			# Label the no_label data
	# 			semi_pred = RNN_model.predict(x_semi_all, batch_size=1024, verbose=True);

	# 			x_semi, y_semi = RNN.get_semi_data(x_semi_all, semi_pred, 0.02);

	# 			x = np.concatenate((x_train, x_semi));

	# 			y = np.concatenate((y_train, y_semi));


	# 			history = RNN_model.fit(x=x, y=y, validation_data=(x_valid, y_valid), batch_size=256, epochs=1, callbacks=[checkpoint]);

	# 		if os.path.exists(save_path):
	# 			RNN_model.load_weights(save_path);