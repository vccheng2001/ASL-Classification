
import imghdr
import cv2 as cv
import numpy as np
import string 
import os
import matplotlib.pyplot as plt


def extract_ORB(img):
    # Check if grayscale; if not, then make it gray
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Compute the descriptors and their keypoints
    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def extract_SIFT(img):
    # Check if grayscale; if not, then make it gray
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Compute the descriptors and their keypoints
    sift = cv.SIFT_create()
    keypoints = sift.detect(img, None)
    return keypoints


def extract_FAST(img):
    # Check if grayscale; if not, then make it gray
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Compute the descriptors and their keypoints
    fast = cv.FastFeatureDetector_create()
    keypoints = fast.detect(img, None)
    return keypoints


def extract_corners(img, num_corners=5,quality=0.05,min_dist=10):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(img, num_corners, quality, min_dist)
    return corners

if __name__ == '__main__':
    path = "../data/Train/A/267_A.jpg"
    img = cv.imread(path)
    orb_kp, orb_ds = extract_ORB(img)
    print(orb_kp)
    print(orb_ds)
    img2 = cv.drawKeypoints(img, orb_kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2)
    plt.show()

    # img = cv.imread(path)
    fast_kp = extract_FAST(img)
    print(fast_kp)
    # print(orb_ds)
    img2 = cv.drawKeypoints(img, fast_kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2)
    plt.show()


    corners = extract_corners(img, 15, 0.01, 1)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),1,255,-1)
    # img2 = cv.drawKeypoints(img, fast_kp, None, color=(0,255,0), flags=0)
    plt.imshow(img)
    plt.show()


