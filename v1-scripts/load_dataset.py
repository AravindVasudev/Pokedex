#!/usr/bin/env python3
'''
This script checks if the dataset exist, if yes, returns the loaded dataset else
creates the `dataset.pickle`, loads it and returns it.
'''

import pickle
from create_dataset_object import dump_dataset

# returns the dataset object
def load_dataset():
    pickleFile = None
    try:
        pickleFile = open('dataset.pickle', 'rb') # open the pickled file
    except FileNotFoundError as err:
        dump_dataset() # Creates the pickle file
        pickleFile = open('dataset.pickle', 'rb') # open the pickled file
    finally:
        dataset = pickle.load(pickleFile) # get the deserialized object
        pickleFile.close() # close the file
        return dataset # return the object


if __name__ == '__main__': load_dataset()
