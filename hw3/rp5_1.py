# hw3 report --- 5
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import random
import csv
import pandas as pd
import sys
from keras.models import load_model, Model
from keras import backend as K
from scipy.misc import imsave
from vis.utils import utils
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf



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

file = open("train.csv", "rb");
reader = pd.read_csv(file);
rawdata = reader.as_matrix();
file.close();
nums_train = rawdata.shape[0];
x_all = np.zeros((rawdata.shape[0], 48*48));
y_all = np.zeros((rawdata.shape[0], ));
for i in range(rawdata.shape[0]):
    x_all[i] = rawdata[i,1].split();
    y_all[i] = rawdata[i,0];
x_all = x_all.astype('float32');
x_all = x_all / 255;
x_all = x_all.reshape((nums_train, 48, 48, 1)); # 48*48 image size; 1 -> gray scale
np.random.seed(100);
x_train, y_train, x_valid, y_valid = validation(x_all, y_all, 0.9);
y_train = np_utils.to_categorical(y_train, 7); # to one hot code
pic = x_valid[10];

print('--- Load Data Success ---')


# 要研究的那層的名字
layer_name = 'conv2d_3'
model = load_model('8.h5py');
layer_idx = utils.find_layer_idx(model, layer_name);
layer_model = Model(inputs=model.input, outputs=model.layers[layer_idx].output);

pic = np.expand_dims(pic, axis=0);
features = layer_model.predict(pic);
print(features.shape);

n = 64;
fig = plt.figure(figsize=(14,6)); #最後圖大小，自己調
for i in range(63):
    if i == 0:
        img = pic[...,0];
        ax = fig.add_subplot(n//16,16,i+1); # (row, col, 放到第幾個位置)
        ax.imshow(img[0,...], cmap='gray'); # cmap 換掉
        # plt.xlabel(lss)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout();
    else: 
        img = features[...,i]
        ax = fig.add_subplot(n//16,16,i+1); # (row, col, 放到第幾個位置)
        ax.imshow(img[0,...], cmap='PuRd'); # cmap 換掉
        # plt.xlabel(lss)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout();
plt.savefig('5_2.png')