#!/usr/bin/env python3

'''
This script reads all the images in `../dataset` and converts them into 40000D
numpy array, label those datapoints, adds them to the datset and dumps the
dataset object using Pickle.
'''

import os
import pickle
import scipy.ndimage
import numpy as np

# A class structure to hold the dataset. Object of this will be dumped using Pickle
class Dataset:
    x = []
    y = []

dataset = Dataset()

def dump_dataset():
    # Iterate through each file in the dataset
    for root, dirs, files in os.walk('../dataset/'):
        for f in files:
            tempRow = scipy.ndimage.imread(os.path.join(root, f)).flatten() # Read the flattened image
            tempLabel = root.split('/')[2] # Extract the directory name for label

            # add to the dataset
            dataset.x.append(tempRow)
            dataset.y.append(tempLabel)

    # Convert X & Y to numpy array
    dataset.x = np.array(dataset.x)
    dataset.y = np.array(dataset.y)

    # Dump the dataset object
    with open('dataset.pickle', 'wb') as saveFile:
        pickle.dump(dataset, saveFile)

    print('Done! Dumped `../dataset/` to `dataset.pickle`.')

if __name__ == '__main__': dump_dataset()
