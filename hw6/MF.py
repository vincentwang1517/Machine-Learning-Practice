# python3 this.py model
# no.86

import sys, io, os
import numpy as np
import pandas as pd

from keras import regularizers
from keras.models import Model
from keras.layers import Input, Embedding, Dense, merge, Flatten, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from mltools import mlTools

class mlTools():

	def _shuffle(x, y, z):
		randomize = np.arange(len(x));
		np.random.shuffle(randomize);
		return x[randomize], y[randomize], z[randomize];

	def validation(x, y, z, percentage):
		all_data_size = len(x);
		train_data_size = int(all_data_size * percentage);
		x_all, y_all, z_all = mlTools._shuffle(x,y,z);

		x_train, y_train, z_train = x_all[0:train_data_size], y_all[0:train_data_size], z_all[0:train_data_size];
		x_valid, y_valid, z_valid = x_all[train_data_size:], y_all[train_data_size: ], z_all[train_data_size: ];
		return x_train, y_train, z_train, x_valid, y_valid, z_valid
class MF():

	def read_data(inputdata):

		data = pd.read_csv(inputdata).as_matrix();
		user = data[:,1];
		movie = data[:,2];
		rating = data[:,3];
		return user, movie, rating;

	def normalize(x):
		x = np.array(x);
		mean = np.mean(x);
		std = np.std(x);
		return (x-mean)/std;

	def model_structure(factors):

		user_input = Input(shape=(1,), name='User');
		user_embedding = Embedding(6040+1, factors, name='User-Embedding', embeddings_initializer='glorot_uniform', dropout=0.3)(user_input);		
		user_embedding = Flatten(name='User-Flatten')(user_embedding);

		user_bias = Embedding(6040+1, 1, name='User-Bias', embeddings_initializer='glorot_uniform')(user_input);
		user_bias = Flatten(name='User-Bias-Flatten')(user_bias);

		movie_input = Input(shape=(1,), name='Movie');
		movie_embedding = Embedding(3952+1, factors, name='Movie-Embedding', embeddings_initializer='glorot_uniform', dropout=0.3)(movie_input);
		movie_embedding = Flatten(name='Movie-Flatten')(movie_embedding);

		movie_bias = Embedding(3952+1, 1, name='Movie-Bias', embeddings_initializer='glorot_uniform')(movie_input);
		movie_bias = Flatten(name='Movie-Bias-Flatten')(movie_bias);

		prod = merge([user_embedding, movie_embedding], mode='dot', name='DotProduct');
		# prod = merge([prod, user_bias, movie_bias], mode='Add', name='Add-Bias');
		prod = Add()([prod,user_bias,movie_bias])

		model = Model([user_input, movie_input], prod);
		adam = Adam(lr=0.0005);
		model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'] );

		print(model.summary());
		return model;

if __name__ == '__main__':

	inputdata = 'train.csv';

	user, movie, rating = MF.read_data(inputdata);

	user, movie, rating, user_va, movie_va, rating_va = mlTools.validation(user, movie, rating, 0.9);

	train = {};
	train['user'] = user;
	train['movie'] = movie;
	train['rating'] = rating;

	valid = {};
	valid['user'] = user_va;
	valid['movie'] = movie_va;
	valid['rating'] = rating_va;

	# print(len(valid['user']))
	model = MF.model_structure(25);


	save_path = 'model/' + sys.argv[1] + 'best.h5';
	checkpoint = ModelCheckpoint(filepath=save_path, 
								verbose=0,
								save_best_only=True,
								monitor='val_acc',
								mode='max')
	model.fit( [train['user'], train['movie']], train['rating'],
				validation_data=([valid['user'], valid['movie']], valid['rating']),
				batch_size=128, epochs=50, callbacks=[checkpoint])
	# model.fit( [train['user'], train['movie']], train['rating'],batch_size=128, epochs=20)