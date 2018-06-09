import numpy as np
from numpy import random

class mlTools():

	def _shuffle(x, y):
		randomize = np.arange(len(x));
		np.random.shuffle(randomize);
		return x[randomize], y[randomize];

	def validation(x, y, percentage):
		all_data_size = len(x);
		train_data_size = int(all_data_size * percentage);
		x_all, y_all = mlTools._shuffle(x,y);

		x_train, y_train = x_all[0:train_data_size], y_all[0:train_data_size];
		x_valid, y_valid = x_all[train_data_size:], y_all[train_data_size:];
		return x_train, y_train, x_valid, y_valid