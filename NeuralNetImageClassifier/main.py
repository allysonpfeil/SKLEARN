import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

import joblib
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier



data = dict()
data['description'] = "10 x 10 image either black or white"
data['label'] = []
data['filename'] = []
data['data'] = []  


directory = 'IMAGES'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    

    im = imread(f)

    data['label'].append(filename)
    data['filename'].append(filename)
    data['data'].append(im)

labels = np.unique(data['label'])
fig, axes = plt.subplots(1, len(labels))
fig.set_size_inches(15,4)
fig.tight_layout()

for ax, label in zip(axes, labels):
    idx = data['label'].index(label)
     
    ax.imshow(data['data'][idx])
    ax.axis('off')
    ax.set_title(label)


X = np.array(data['data'])
y = np.array(data['label'])

new_x = []

for item in X:
    FLAT = [item for sublist in item for item in sublist]
    new_x.insert(-1, FLAT)

    
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(new_x, y)




