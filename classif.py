import json  # To deal with json labels file
import numpy as np  # For arrays
import matplotlib.pyplot as plt  # To plot things
from PIL import Image  # To deal with images
import os  # To deal with directories
from tqdm import tqdm  # For progress bar
import sys  # To exit program
import pickle as pkl
import h5py
import pandas as pd  # To deal with dataframes
from sklearn.model_selection import StratifiedShuffleSplit  # To have evenly splitted training set and test set without unbalancing the dataset
# from imblearn.over_sampling import SMOTE  # For data augmentation


def process_data(width=512, height=512, test_size=0.3):
    print('Creating a csv from data...')
    with open('Datasets/labels.json') as f:
        labels = dict(json.load(f))
    files = os.listdir('Datasets/Images')
    y = np.zeros(len(files), dtype='uint8')
    X = np.zeros((len(files), height, width, 3), dtype='uint8')
    for i, j in tqdm(enumerate(files), total=len(files)):
        im = Image.open('Datasets/Images/'+j)
        im = im.resize((width, height))
        y[i] = labels[j]
        X[i] = im
    print('Splitting evenly the dataset...')
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=20)
    for train_idx, test_idx in split.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    print('Saving it...')
    file = h5py.File('Datasets/data.h5', 'w')
    file.create_dataset(
        "X_train", np.shape(X_train), dtype='uint8', data=X_train, compression="lzf"
    )
    file.create_dataset(
        "X_test", np.shape(X_test), dtype='uint8', data=X_test, compression="lzf"
    )
    file.create_dataset(
        "y_train", np.shape(y_train), dtype='uint8', data=y_train, compression="lzf"
    )
    file.create_dataset(
        "y_test", np.shape(y_test), dtype='uint8', data=y_test, compression="lzf"
    )
    file.close()
    print('Done.')


def load_data():
    print('Loading the data...')
    file = h5py.File('Datasets/data.h5', 'r+')
    X_train = np.array(file["/X_train"])
    X_test = np.array(file["/X_test"])
    y_train = np.array(file["/y_train"])
    y_test = np.array(file["/y_test"])
    file.close()
    print('Done.')
    return X_train, y_train, X_test, y_test


def plot_sample(X, y):
    _, ax = plt.subplots(nrows=1, ncols=5)
    for i in range(5):
        ax[i].imshow(np.array(X[np.where(y == i)[0]][0]).reshape((512, 512, 3)))
        ax[i].axis('off')
        ax[i].set_title('Stage %d' % (i))
    plt.show()


if __name__ == "__main__":
    # process_data()
    X_train, y_train, X_test, y_test = load_data()
    plot_sample(X_train, y_train)
