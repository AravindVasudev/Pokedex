import os
from PIL import Image

for root, dirs, files in os.walk('../dataset/'):
    for f in files:
        print(f)
