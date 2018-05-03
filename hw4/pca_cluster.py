# hw4 --- PCA & clustering

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import csv
import sys

pca_component = 400; # 300, 400, 150

x = np.load(sys.argv[1]);	# x.shape = (140000,784)
x = x.astype("float32") / 255.;
pca = PCA(n_components=pca_component, copy=True, whiten=True);
z = pca.fit_transform(x);
print("---- PCA Success ----")

# xbar = np.mean(x, axis=0);	# xbar.shape = (784, )

# xbar = np.tile(xbar, (x.shape[0], 1));	# xbar.shape = (140000, 784)

# x = x - xbar;

# U, s, V = np.linalg.svd(x, full_matrices=False); # s: automatically sorted
# # x = USV* ---> eigenvectors are V[0], V[1], V[2]......and sorted.
# # V[0].shape = (784, )
# # print(U.shape, s.shape, V.shape);

# dim = 20;
# W = np.zeros((dim, 784)); # transpose already
# for i in range(dim):
# 	W[i] = V[i];
# # print(W.shape);

# Z = np.dot(x, W.T);		# Z.shape = (20,140000)
# # print(Z.shape)         

kmeans = KMeans(n_clusters=2,random_state=0).fit(z);
# np.save(sys.argv[1], kmeans.labels_);
print("---- Kmeans Success ----")

file = open(sys.argv[2], "rb");
reader = pd.read_csv(file)
x_test = reader.as_matrix();
file.close();

print("---- Start to predict ----");
file1 = open(sys.argv[3], "w+");
result = csv.writer(file1, delimiter=',', lineterminator='\n')
result.writerow(["ID", "Ans"]);
for i in range(x_test.shape[0]):
	img1 = x_test[i][1];
	img2 = x_test[i][2];
	if (kmeans.labels_[img1] == kmeans.labels_[img2]):
		result.writerow([i, 1]);
	else:
		result.writerow([i, 0]);
file1.close();