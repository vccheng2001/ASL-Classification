import cv2 as cv
import numpy as np
import string 
import os

''' Given an image, extract its contours '''
def extract_contours(img):
    h, w, = img.shape 

    src = cv.resize(img, dsize = (w*4,h*4))

    # cv.imshow('img', src)
    # cv.waitKey(0)

    # convert img into binary
    _, bw = cv.threshold(src, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # calculating Contours
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            continue
        # Draw each contour only for visualisation purposes
        cv.drawContours(src, contours, i, (0, 0, 255), 2)
         
    cv.imshow('contours', src)
    cv.waitKey(0)



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
