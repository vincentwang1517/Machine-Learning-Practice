# python3 this.py model.h5 output.csv

import sys, io, os, csv
import numpy as np
import pandas as pd

from keras.models import load_model
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Embedding, Dense, merge, Flatten, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def read_test_data():

	inputfile = sys.argv[1];

	data = pd.read_csv(inputfile).as_matrix();

	user = data[:,1];
	movie = data[:,2];

	return user, movie;

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

# def read_data(inputdata):
# 		data = pd.read_csv(inputdata).as_matrix();
# 		user = data[:,1];
# 		movie = data[:,2];
# 		rating = data[:,3];
# 		return user, movie, rating;

if __name__ == '__main__':

	user, movie = read_test_data();
	print(user)

	modelname = 'rp1_100best.h5';
	model = load_model(modelname);

	print('--- Predict ---');
	pred = model.predict([user, movie]);

	inputfile = 'train.csv'
	# user_train, movie_train, rating_train = read_data(inputfile);
	# mean = np.mean(rating_train);
	# std = np.std(rating_train);
	meanstd = np.load('meanstd.npy');
	mean = meanstd[0];
	std = meanstd[1];
	pred = pred * std + mean;


	print()
	outputfile = sys.argv[2];
	result = csv.writer(open(outputfile, 'w+'), delimiter = ',', lineterminator = '\n');
	result.writerow(['TestDataID', 'Rating']);
	for i in range(len(pred)):
		result.writerow(('%d' %(i+1), pred[i][0])) ;