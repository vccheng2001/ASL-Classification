from ctypes import c_void_p
import numpy as np
import dataloader
import argparse
import cv2 as cv
import matplotlib.pyplot as plt

''' Given X, a matrix where each column vector represents an image, compute the first d principle components'''
def pca(X, d=100):
    avgX = np.average(X, axis=0)
    # avgX = np.repeat(avgX, X.shape[1], axis=1)
    centeredX = np.subtract(X, avgX)
    covX = np.cov(centeredX, rowvar=False)
    eigVals, eigVecs = np.linalg.eig(covX)
    idx = eigVals.argsort()[::-1]
    # sortedEigVals = eigVals[idx]
    sortedEigVecs = eigVecs[:, idx]
    return sortedEigVecs[:, :d]



def main():
    parser = argparse.ArgumentParser(description="Compute PCA on image data.")
    parser.add_argument("--num_components", type=int, help="Number of principle components to store")
    parser.add_argument("--path", help="Path to the dataset")
    parser.add_argument("--dest_path", help="File destination path to be saved to")
    parser.add_argument("--dest_name", help="File name to be saved under")
    args = parser.parse_args()

    imgs, labels = dataloader.load_data(args.path, gray=True)
    if args.num_components is not None:
        P = pca(imgs, args.num_components)
    else:
        P = pca(imgs)

    print(P.shape)
    
    if args.dest_name is not None:
        if args.dest_path is None:
            dataloader.save_features(P, args.dest_name, "")
        else:
            dataloader.save_features(P, args.dest_name, args.dest_path)

if __name__ == '__main__':
    main()