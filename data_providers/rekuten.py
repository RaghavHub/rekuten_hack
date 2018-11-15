import tempfile
import os
import scipy.io
import cv2
import glob
import csv
from scipy import misc

import numpy as np

from .base_provider import ImagesDataSet, DataProvider
# from .downloader import download_data_url


class RekutenDataSet(ImagesDataSet):
    n_classes = 37

    def __init__(self, images, labels, shuffle, normalization):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            shuffle: `bool`, should shuffle data or not
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
        """
        self.shuffle = shuffle
        self.images = images
        self.labels = labels
        self.normalization = normalization
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle:
            self.images, self.labels = self.shuffle_images_and_labels(
                self.images, self.labels)

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.images[start: end]
        labels_slice = self.labels[start: end]
        # due to memory error it should be done inside batch
        if self.normalization is not None:
            images_slice = self.normalize_images(
                images_slice, self.normalization)
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice


class RekutenDataProvider(DataProvider):
    def __init__(self, save_path=None, validation_set=None,
                 validation_split=None, shuffle=False, normalization=None,
                 one_hot=True, **kwargs):
        """
        Args:
            save_path: `str`
            validation_set: `bool`.
            validation_split: `int` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `bool`, should shuffle data or not
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            one_hot: `bool`, return lasels one hot encoded
        """
        self._save_path = save_path
        train_images = []
        train_labels = []
        for part in ['train', 'extra']:
            images, labels = self.get_images_and_labels(part, one_hot)
            # print "shape",images.shape
            train_images.append(images)
            train_labels.append(labels)
        train_images = np.vstack(train_images)
        # print "after shape",len(train_images)
        if one_hot:
            train_labels = np.vstack(train_labels)
        else:
            train_labels = np.hstack(train_labels)
        if validation_set and validation_split:
            rand_indexes = np.random.permutation(train_images.shape[0])
            valid_indexes = rand_indexes[:validation_split]
            train_indexes = rand_indexes[:validation_split]
            valid_images = train_images[valid_indexes]
            valid_labels = train_labels[valid_indexes]
            train_images = train_images[train_indexes]
            train_labels = train_labels[train_indexes]
            self.validation = RekutenDataSet(
                valid_images, valid_labels, shuffle, normalization)

        self.train = RekutenDataSet(
            train_images, train_labels, shuffle, normalization)

        test_images, test_labels = self.get_images_and_labels('test', one_hot)
        self.test = RekutenDataSet(test_images, test_labels, False, normalization)

        if validation_set and not validation_split:
            self.validation = self.test

    def get_images_and_labels(self, name_part, one_hot=False):
        # print name_part
        size = 100
        ratio = [7,2,1]
        sum = ratio[0] +ratio[1] +ratio[2]
        split1 = int(float(ratio[0]) / float(sum) * size)
        split2 = int(float(ratio[1]) / float(sum) * size)
        split3 = int(float(ratio[2]) / float(sum) * size)
        print "splits", split1,split2,split3
        img = np.ndarray(shape=(size, 180, 180, 3))
        receipe_list = []
        cat_list = []
        lbl = np.ndarray(shape=(size, ),dtype=int)
        i = 0
        dir_path = "../../training-data/train.csv"
        with open(dir_path, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                if i >= size -1:
                    break;
                array = row[0].split(',')
                receipe = array[0].split('.')[0]
                receipe_list.append(receipe)
                cat_list.append(array[1])
                i = i + 1
        i = 0
        # print cat_list
        for receipe in receipe_list:
            myFile = "../../training-data/train-images/" + str(receipe) + ".jpg"
            image = cv2.imread(myFile)
            image = misc.imresize(image, (180, 180))
            img[i] = image
            lbl[i] = cat_list[receipe_list.index(receipe)][0]
            i = i + 1
        # labels = lbl.reshape((-1))
        if one_hot:
             lbl = self.labels_to_one_hot(lbl)
        if name_part == 'train':
            images = img[:split1, :, :]
            labels = lbl[:split1]
            # print "Image: ", images.shape
            # print "Label: ", labels.shape
            # print "Label: ", labels
            return images, labels
        elif name_part == 'extra':
            images = img[split1:(split1+split2), :, :]
            labels = lbl[split1:(split1+split2)]
            # print "Images: ", images
            # print "Label: ", labels
            # print "Image: ", images.shape
            # print "Label: ", labels.shape
            return images, labels
        else:
            images = img[-split3:, :, :]
            labels = lbl[-split3:]
            # print "Image: ", images.shape
            # print "Label: ", labels.shape
            return images, labels

    @property
    def n_classes(self):
        return 37

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(tempfile.gettempdir(), 'Rekuten')
        return self._save_path

    @property
    def data_url(self):
        return "http://ufldl.stanford.edu/housenumbers/"

    @property
    def data_shape(self):
        return (180, 180, 3)


if __name__ == '__main__':
    # WARNING: this test will require about 5 GB of RAM
    # import matplotlib.pyplot as plt
    #
    # def plot_images_labels(images, labels, axes, main_label):
    #     plt.text(0, 1.5, main_label, ha='center', va='top',
    #              transform=axes[len(axes) // 2].transAxes)
    #     for image, label, axe in zip(images, labels, axes):
    #         axe.imshow(image)
    #         axe.set_title(np.argmax(label))
    #         axe.set_axis_off()
    #
    # n_plots = 10
    # fig, axes = plt.subplots(nrows=2, ncols=n_plots)
    #
    dataset = RekutenDataProvider()
    # plot_images_labels(
    #     dataset.train.images[:n_plots],
    #     dataset.train.labels[:n_plots],
    #     axes[0],
    #     'Original dataset')
    #
    dataset = RekutenDataProvider(shuffle=True)
    # plot_images_labels(
    #     dataset.train.images[:n_plots],
    #     dataset.train.labels[:n_plots],
    #     axes[1],
    #     'Shuffled dataset')
    #
    # plt.show()
