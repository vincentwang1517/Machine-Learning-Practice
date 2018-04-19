
import numpy as np
import csv
import pandas as pd
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU, PReLU
import tensorflow as tf

def model_structure():
	model = Sequential();
	
	model.add(Conv2D( filters=64, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform', input_shape=(48,48,1)));
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.27)); #
	model.add(Conv2D( filters=64, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); #
	model.add(LeakyReLU(alpha=0.1));	#
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2)));
	model.add(Dropout(0.2));	

	model.add(Conv2D( filters=128, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.25));
	model.add(Conv2D( filters=128, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2))); # 128 x 12 x 12
	model.add(Dropout(0.35));

	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.25));
	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.25));
	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2))); # 64 x 6 x 6
	model.add(Dropout(0.25));

	model.add(Flatten());
	model.add(Dense(256,  kernel_initializer='random_uniform'));
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(Dropout(0.44));
	model.add(Dense(256, kernel_initializer='random_uniform'));
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(Dropout(0.37));
	model.add(Dense(7, activation='softmax'));

	print(model.summary());
	return model;

file = open(sys.argv[1], "rb");
reader = pd.read_csv(file);
rawdata = reader.as_matrix();
file.close();

nums_test = rawdata.shape[0];
x_test = np.zeros((rawdata.shape[0], 48*48));

for i in range(rawdata.shape[0]):
	x_test[i] = rawdata[i,1].split();

x_test = x_test.astype('float32');
x_test = x_test / 255;
x_test = x_test.reshape((nums_test, 48, 48, 1)); # 48*48 image size; 1 -> gray scale

# model = load_model(sys.argv[3]);
model = model_structure();
model.load_weights(sys.argv[3]);

y_predict = model.predict_classes(x_test);

result = csv.writer(open(sys.argv[2], 'w+'), delimiter = ',', lineterminator = '\n');
result.writerow(['id', 'label'])
for i in range(len(y_predict)):
	result.writerow([(i), y_predict[i]] );