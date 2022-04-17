import imghdr
import cv2 as cv
import numpy as np
import string 
import os


''' 
Histogram of Oriented Gradients
------------------------------------
To compute HOGs we create a histogram for each cell of an image patch
and then normalize over the patch.

Divide image into cells of mxn pixels 
Normalization across blocks, each of which contains several cells 
'''
def extract_HOG(img):

    winSize = (28,28) # size of sliding window run across image 
    blockSize = (16,16) # block_size is the number of cells which fit in the patch
    blockStride = (1,1)
    cellSize = (8,8) # cell_size is the size of the cells of the img patch over which to calculate the histograms
    nbins = 9 # orientation bins
    # signedGradient=T
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

    
    hog_features = hog.compute(img)
    return hog_features



''' Given an image, extract its contours '''
def extract_contours(img):
    h, w, = img.shape 

    # Resize (optional)
    # img = cv.resize(img, dsize = (w*4,h*4))

    # cv.imshow('img', img)
    # cv.waitKey(0)

    # convert img into binary
    _, bw = cv.threshold(img, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # calculating Contours
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            continue
        # Draw each contour only for visualisation purposes
        cv.drawContours(img, contours, i, (0, 0, 255), 2)
         
    cv.imshow('contours', img)
    cv.waitKey(0)



''' Given an image, extract its ORB Descriptions'''
def extract_ORB(img):
    # Check if grayscale; if not, then make it gray
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Compute the descriptors and their keypoints
    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors



''' Need to install the opencv-contrib-python library first'''
def extract_SIFT(img):
    # Check if grayscale; if not, then make it gray
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Compute the descriptors and their keypoints
    sift = cv.SIFT_create()
    keypoints = sift.detectAndCompute(img, None)
    return keypoints



def extract_FAST(img):
    # Check if grayscale; if not, then make it gray
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Compute the descriptors and their keypoints
    fast = cv.FastFeatureDetector_create()
    keypoints = fast.detect(img, None)
    return keypoints



def extract_corners(img, num_corners=5,quality=0.01,min_dist=10):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(img, num_corners, quality, min_dist)
    return corners



def main():
    ALPHABET = string.ascii_uppercase
    for char in ALPHABET: 
        print(f'Processing letter {char}')

        path = f'dataset/Train/{char}'

        # check if dir exists 
        if not os.path.isdir(path): continue 

        # process one image
        file  = os.listdir(path)[0]

        img = cv.imread(os.path.join(path, file), 0)
        features = extract_contours(img)


if __name__ == "__main__":
    main()
