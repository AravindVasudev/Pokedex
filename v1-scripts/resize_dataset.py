#!/usr/bin/env python3

'''
This script reads all the images in `../dataset` folder, crops them to
`200 x 200` and converts them to grayscale.
'''

import os
from PIL import Image, ImageOps

# Iterate through each file in the dataset
for root, dirs, files in os.walk('../dataset/'):
    for f in files:
        imgF = Image.open(os.path.join(root, f)).convert('RGB') # Open the file in RGB mode
        imgF = ImageOps.fit(image=imgF, size=(32, 32), method=Image.ANTIALIAS) # Crop the image
        imgF = ImageOps.grayscale(imgF) # apply grayscale
        imgF.save(os.path.join(root, f)) # save the image
        print(os.path.join(root, f), '- Done!') # logs for each image

print('Done! Resized all images to 200 x 200 and to grayscale')
