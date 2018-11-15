import cv2
import glob
import csv
from scipy import misc

import numpy as np

def labels_to_one_hot(labels):
    """Convert 1D array of labels to one hot representation

    Args:
        labels: 1D numpy array
    """
    n_classes = 37
    new_labels = np.zeros((labels.shape[0], n_classes))
    new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
    return new_labels

name_part = 'test'

size = 100
ratio = [7, 2, 1]
sum = ratio[0] + ratio[1] + ratio[2]
split1 = int(float(ratio[0]) / float(sum) * size)
split2 = int(float(ratio[1]) / float(sum) * size)
split3 = int(float(ratio[2]) / float(sum) * size)
print "splits", split1, split2, split3
img = np.ndarray(shape=(size, 180, 180, 3))
receipe_list = []
cat_list = []
lbl = np.ndarray(shape=(size,), dtype=int)
i = 0
dir_path = "../../../training-data/train.csv"
with open(dir_path, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        if i >= size - 1:
            break;
        array = row[0].split(',')
        receipe = array[0].split('.')[0]
        receipe_list.append(receipe)
        cat_list.append(array[1])
        i = i + 1
i = 0
# print cat_list
for receipe in receipe_list:
    myFile = "../../../training-data/train-images/" + str(receipe) + ".jpg"
    image = cv2.imread(myFile)
    image = misc.imresize(image, (180, 180))
    img[i] = image
    lbl[i] = cat_list[receipe_list.index(receipe)][0]
    i = i + 1
# labels = lbl.reshape((-1))
if True:
    lbl = labels_to_one_hot(lbl)
    print "shape",lbl.shape
if name_part == 'train':
    images = img[:split1, :, :]
    labels = lbl[:split1]
    print images, labels
elif name_part == 'extra':
    images = img[split1:split2, :, :]
    labels = lbl[split1:split2]
    print images, labels
else:
    images = img[split2:, :, :]
    labels = lbl[split2:]
    print images, labels