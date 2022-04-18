from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import dataloader

if __name__ == "__main__":
    with open("test_pca.npy", "rb") as f:
        P = np.load(f)

    imgs, labels = dataloader.load_data("../data/Test", gray=True)
    # test_image = cv.imread("../data/Test/H/1304_H.jpg")
    # print(test_image.shape)
    # test_image = test_image.flatten()
    print(P.shape)
    print(imgs.shape)
    reduced = imgs @ P
    recon_image = P @ reduced[1, :]
    print(labels[1])
    print(recon_image)

    plt.imshow(P[:, 1].reshape(28, 28), cmap='gray')
    plt.show()

    plt.imshow(recon_image.reshape(28, 28), cmap='gray')
    plt.show()

    
