<h1 align="center">Detecting Eye Diseases Thanks to Machine Learning</h1>

# Introduction

Nowadays, data science represents a non-negligible help for physicians for them to be faster and more accurate in their task of detecting diseases. Plus, it is particularly useful in places of the world where there is no trained physicians to assess people. Eye diseases are generally hard to identify before reaching a certain level of development. Diabetic Retinopathy is one of these.  

<img src="Images/DiabeticRetinopathy.png">

The purpose of this project is to train different learning models on a database composed of eyes pictures, some of which are ill at different stages. Here is a sample of the database, from stage 0 (not ill) to stage 4 (proliferative diabetic retinopathy):

<img src="Images/Stages.png">

It seems very hard for an average person to detect the disease by looking at these images, and that is why we are going to implement models to do it for us.

# The Database

We were given a folder of images with precise names and a json folder contraining a list of tuples with the name of the image file as first element and the label as second element.
The first problem is the fact that not all images are the same size. I resized them to have a 512x512 image for every image and dropped four images whose sizes were less than 512x512.  
Secondly, if we get back to the database sample, there can be great contrast differences between images which will be an issue for our model:
<img src="Images/DatasetContrast.png">
Finally, we need a training set on which we will train the model and a testing set on which we will test it. We will take 70% of the dataset as the training set and 30% of it as the testing set making sure we keep the same proportion of labels in both sets.  
The following function is doing all of this:

```python
import json  # To deal with json labels file
import pickle as pkl # To save and load data
def process_data(width=512, height=512, test_size=0.3):
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
```
And to load this new processed and saved data:
```python
def load_data():
    file = h5py.File('Datasets/data.h5', 'r+')
    X_train = np.array(file["/X_train"])
    X_test = np.array(file["/X_test"])
    y_train = np.array(file["/y_train"])
    y_test = np.array(file["/y_test"])
    file.close()
    return X_train, y_train, X_test, y_test
```




SVM, LASSO, CNN, FFN