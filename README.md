# Pokedex
A Pokémon Image Classifier using EigenFaces(PCA) algorithm. In this, a dataset of
20 pokémon, each with 30 images(600 sample images) are taken, resized to 200 x 200,
grayscaled and a classifier is trained to this dataset using the dataset's Principle
Components. The train set and the test set split is done using KFolds method(4 splits) and the classifier is evaluated using various metrics.

### Scripts Description

|        Name         | Description |
|---------------------|-------------|
| `resize_dataset.py` | This script reads all the images in `../dataset` folder, crops them to`200 x 200` and converts them to grayscale. |
| `create_dataset_object.py` | This script reads all the images in `../dataset` and converts them into 40000D numpy array, label those datapoints, adds them to the datset and dumps the dataset object using Pickle. |
| `load_dataset.py` | This script checks if the dataset exist, if yes, returns the loaded dataset else creates the `dataset.pickle`, loads it and returns it. |
| `create_classifier.py` | This script creates the classifier, trains it, and dumps it to `classifier.pickle` file. |

### TODOs
  - [ ] Obtain a cleaner dataset
  - [ ] Improve Accuracy
  - [ ] Add a Interface to test new images.
