# hw4 - 1 pca of colored face
import sys
import os
import numpy as np
from skimage import io

# img = io.imread("./Aberdeen/*.jpg");
# print(img.shape);
strr = sys.argv[1] + "/*.jpg";
coll = io.ImageCollection(strr);
coll = np.array(coll);		# (415, 600, 600, 3)
coll = coll.flatten().reshape(415, 600*600*3);	# (415, 1080000)
coll = coll.astype("float64");

# rp1.1
meann = np.mean(coll, axis = 0);

# rp1.2
nb_eigenface = 4;
x_ = coll - np.tile(meann, (415, 1));
U, s, V = np.linalg.svd(x_.T, full_matrices=False);		# U (1080000, 415)
eigenvector = [];
for i in range(nb_eigenface):
	eigenvector.append(U[:, i]);		
eigenvector = np.array(eigenvector);	# eigenvector (4,1080000)

# rp 1.3
strr2 = os.path.join(sys.argv[1], sys.argv[2]);
imgin = io.imread(strr2);
imgin = imgin.flatten().astype("float64");
imgin -= meann;

z = np.dot(eigenvector, imgin);
img = np.zeros((1,600*600*3));
for j in range(len(z)):
	img += z[j]*eigenvector[j];
img += meann;# img (1, 600*600*3)

img -= np.min(img);
img /= np.max(img);
img = (img*255).astype(np.uint8).reshape(600,600,3);
io.imsave(sys.argv[3], img);