# hw3 report --- 5
import matplotlib
matplotlib.use('Agg')
import numpy as np
from keras.models import load_model
from keras import backend as K
from scipy.misc import imsave
import matplotlib.pyplot as plt

# 要研究的那層的名字
layer_name = 'conv2d_3'

model = load_model('8.h5py');
layer_dict = dict([(layer.name, layer) for layer in model.layers]);



def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

img_width = 48;
img_height = 48;
input_img_ = model.input;
kept_filters = [];

n = 64 # 要研究的filter的數目
for filter_index in range(n):
    layer_output = layer_dict[layer_name].output;
    loss = K.mean(layer_output[...,filter_index]);
    grads = K.gradients(loss, input_img_)[0];
    grads = normalize(grads);
    # grads = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img_], [loss, grads])

    input_img = np.random.random((1, 48, 48, 1)) * 20 + 128.;
    step = 0.05; # 自己調
    for i in range(50):    
        loss_value, grads_value = iterate([input_img]);
        input_img += grads_value * step;

    img = deprocess_image(input_img[0]);
    kept_filters.append((img, loss_value))  
    print('Filter %d processed ' % (filter_index))


# we will stich the best 64 filters on a 8 x 8 grid.
# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
# kept_filters.sort(key=lambda x: x[1], reverse=True)
# kept_filters = kept_filters[:n * n]


fig = plt.figure(figsize=(14,6)); #最後圖大小，自己調
for i in range(len(kept_filters)):
    img, lss = kept_filters[i];
    ax = fig.add_subplot(n//16,16,i+1); # (row, col, 放到第幾個位置)
    ax.imshow(img[...,0], cmap='PuRd'); # cmap 換掉
    # plt.xlabel(lss)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout();
plt.savefig('5_123.png')