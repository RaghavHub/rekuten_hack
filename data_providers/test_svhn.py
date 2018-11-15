import tempfile
import os
import scipy.io


import numpy as np


def labels_to_one_hot( labels):
    """Convert 1D array of labels to one hot representation

    Args:
        labels: 1D numpy array
    """
    n_classes = 10
    new_labels = np.zeros((labels.shape[0], n_classes))
    new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
    return new_labels

from downloader import download_data_url

save_path = "../../../svhn_data/"
url = "http://ufldl.stanford.edu/housenumbers/" + 'train' + '_32x32.mat'
download_data_url(url, save_path)
filename = os.path.join(save_path, "train" + '_32x32.mat')
data = scipy.io.loadmat(filename)
print data['X'].shape
print data['y'].shape
images = data['X'].transpose(3, 0, 1, 2)
labels = data['y'].reshape((-1))
labels[labels == 10] = 0
labels = labels_to_one_hot(labels)

print images.shape
print labels.shape