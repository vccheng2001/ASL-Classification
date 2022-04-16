import cv2 as cv
import numpy as np
import string 
import os
from dataloader import load_data
from sklearn.manifold import TSNE
from dataloader import map_char_to_num
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm



def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

''' Load data (see dataloader.py) '''

path = 'dataset/Test'
data, labels = load_data(path, preproc='hog')
data = data.astype(np.double)
labels = labels.astype(np.double)
n_classes = 24 
n_samples, n_features = data.shape

# Data shape: (n_samples * n_features)
# Labels shape: (n_samples, )
print(f'Data shape: {data.shape}, Labels shape: {labels.shape}')


# Take the first 2500 data points
X = data
y = labels

''' t-SNE '''
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)


colors = 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'lightcoral', 'cyan', 'sienna', 'palegreen', 
'indigo', 'aquamarine', 'lime', 'cornflowerblue', 'thistle', 'plum', 'pink', 
'slategray', 'peru', 'salmon', 'lightyellow', 'darkseagreen', 'forestgreen'

plt.figure(figsize=(6, 5))
for i, c, label in zip(map_char_to_num.values(), colors,  map_char_to_num.keys()):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.title('t-SNE on ASL Test Dataset')
plt.legend()
plt.show()