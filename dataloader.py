import numpy as np
import os
import glob
import string
import cv2 
import preprocess
from preprocess import extract_HOG, extract_contours, extract_ORB, extract_FAST, extract_corners

'''
Test dataset should have 7172 total images
Train dataset should have 27,455 total images
'''

# Maps each letter (except J,Z) to int
map_char_to_num = {'A':0,
                   'B':1,
                   'C':2,
                   'D':3,
                   'E':4,
                   'F':5,
                   'G':6,
                   'H':7,
                   'I':8,
                   # skip J
                   'K':9,
                   'L':10,
                   'M':11,
                   'N':12,
                   'O':13,
                   'P':14,
                   'Q':15,
                   'R':16,
                   'S':17,
                   'T':18,
                   'U':19,
                   'V':20,
                   'W':21,
                   'X':22,
                   'Y':23} 
                   # skip Z


''' 
Returns data X, labels y in np array format

Data: image data, shape: (N * n_features)
Labels: map each letter to int, shape: (N,) 

If preproc is not None, then each image is preprocessed (e.g. feature extraction)
'''
def load_data(path, preproc=None, gray=False):

    data = []
    labels = []

    ALPHABET = string.ascii_uppercase
    for char in ALPHABET: 
        if char == "J" or char == "Z": continue

        files = os.listdir(os.path.join(path, char))
        for f in files:
            img = cv2.imread(os.path.join(path, char, f))
            if gray and len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if preproc is None: # just use flattened image as feature 
                data.append(img.flatten()) 
            else:               # use some kind of feature extraction 
                if preproc == "hog":
                    preproc_img = extract_HOG(img)
                    data.append(preproc_img)
                elif preproc == "orb":
                    orb_kp, orb_des = extract_ORB(img)
                    data.append((orb_kp, orb_des))
                elif preproc == "corners":
                    preproc_img = extract_corners(img)
                    data.append(preproc_img)
                elif preproc == "fast":
                    preproc_img = extract_FAST(img)
                    data.append(preproc_img)
           
            labels.append(map_char_to_num[char]) 

    return np.array(data), np.array(labels)




''' 
Returns data, labels 
Data: array of all filenames (e.g. 'dataset/Test/A/2645_A.jpg' )
Labels: array of corresponding letters (e.g. 'A')
'''
def load_data_files(path):

    data = []
    labels = []

    ALPHABET = string.ascii_uppercase
    for char in ALPHABET: 
        # skip J, Z
        if char == "J" or char == "Z": continue

        files = os.listdir(os.path.join(path, char))

        for f in files:
            data.append(os.path.join(path, char, f))
            labels.append(char)

    return data, labels


def save_features(data, name, path):
    if not path.endswith("/") and len(path) > 0:
        fi = path + "/" + name + ".npy"
    else:
        fi = path + name + ".npy"

    with open(fi, "wb") as f:
        np.save(f, data)
    print("Data saved at " + fi)





# load_data('dataset/Test', preproc="hog")