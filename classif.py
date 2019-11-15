import json  # To deal with json labels file
import numpy as np  # For arrays
import matplotlib.pyplot as plt  # To plot things
from PIL import Image  # To deal with images
import os  # To deal with directories
from tqdm import tqdm  # For progress bar
import sys  # To exit program
import pandas as pd  # To deal with dataframes
from sklearn.model_selection import StratifiedShuffleSplit  # To have evenly splitted training set and test set without unbalancing the dataset
from imblearn.over_sampling import SMOTE  # For data augmentation


def create_df_from_data(common_size=(512, 512)):
    print('Creating a csv from data...')
    with open('labels.json') as f:
        labels = dict(json.load(f))
    files = os.listdir('images')
    data = []
    for j in tqdm(files, total=len(files)):
        im = Image.open('images/'+j)
        im = im.resize((common_size))
        data += [{'label': labels[j], 'image': np.array(im)}]

    df = pd.DataFrame(data)
    df.to_pickle('datasets/eyesdata.pkl')
    print('Done.')


def even_split(df, param='label', test_size=0.3):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=20)
    for train_idx, test_idx in split.split(df, df[param]):
        df_train = df.loc[train_idx]
        df_test = df.loc[test_idx]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return [df_train, df_test]


def get_X_y(df, param='label'):
    return df[df.columns.difference([param])], df[[param]]


if __name__ == "__main__":
    create_df_from_data()
    df = pd.read_pickle('datasets/eyesdata.pkl')
    df_train, df_test = even_split(df)
    print(df_train.groupby('label').count())
    df_counts = df_train.groupby('label').count()
    samp_strat = {}
    for i in range(df_counts.shape[0]):
        samp_strat[i] = df_counts.loc[0, 'image']-df_counts.loc[i, 'image']
    sm = SMOTE(sampling_strategy=samp_strat, random_state=42)
    X_res, y_res = sm.fit_resample(get_X_y(df_train))
