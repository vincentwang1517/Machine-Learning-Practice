# Hw3 CNN
# python3 this.py 
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import random
import csv
import pandas as pd
import sys
from keras.models import Sequential, load_model

import matplotlib.pyplot as plt
from matplotlib import cm
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations


file = open("test1.csv", "rb");
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

# ------------------------------- --------------------------
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
model = load_model('8.h5py');
# print(model.summary())
y_pred = model.predict_classes(x_test);

print(y_pred)
print('---- Predict Success ----')


class_idx = 5;
indices = np.where(y_pred[:] == class_idx)[0];
idx = indices[0];
# idx = 3;

name = sys.argv[1];

plt.figure(frameon=False);
plt.imshow(x_test[idx][...,0], cmap='gray');
# plt.title(classes[class_idx], fontsize='large');
plt.savefig(name+'_1.png', bbox_inches='tight');

last_layer_idx = utils.find_layer_idx(model, 'dense_3');
model.layers[last_layer_idx].activation = activations.linear;
model = utils.apply_modifications(model);
grads = visualize_saliency(model, last_layer_idx, filter_indices=class_idx, seed_input=x_test[idx]);
grads = grads.astype('float32');
grads /= 255;
plt.figure();
plt.imshow(grads, cmap='jet');
plt.colorbar();
# plt.title('heatmap', fontsize='large')
# print(cm.jet(0.2))
plt.savefig(name + '_2.png', bbox_inches='tight');




see = x_test[idx].reshape(48,48);
meann = np.mean(see);
# see[np.where(grads[...,1] == 0 and grads[...,0] < 0.3)] = meann;
for i in range(48):
	for j in range(48):
		if grads[i,j,0] != 0 or grads[i,j,1] > 0. :
			continue;
		else: 
			see[i,j] = meann;
plt.figure();
plt.imshow(see, cmap='gray');
plt.colorbar();
# plt.title('masked', fontsize='large');
plt.savefig(name + '_3.png', bbox_inches='tight');