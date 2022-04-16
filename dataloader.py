import numpy as np
import os
import glob
import string 

''' 
Returns data X, labels y
Test dataset should have 7172 total images
Train dataset should have 27,455 total images
'''
def load_data(path):

    data = []
    labels = []

    ALPHABET = string.ascii_uppercase
    for char in ALPHABET: 
        # skip J, Z
        if char == "J" or char == "Z": continue

        files = os.listdir(os.path.join(path, char))

        for f in files:
            data.append(f)
            labels.append(char)

    return data, labels


load_data('dataset/Test')