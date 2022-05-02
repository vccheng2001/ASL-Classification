import numpy as np
import matplotlib.pyplot as plt
import argparse
import dataloader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import cv2 as cv
import preprocess
from skimage import data, exposure
from skimage.io import imread
from skimage.feature import hog
import time

def main():
    parser = argparse.ArgumentParser(description="Do knn on the HOG features of ASL")
    parser.add_argument("--train_path", help="Path to training data")
    parser.add_argument("--test_path", help="Path to test data")
    parser.add_argument("--use_saved", type=int, help="Used the save npy files")
    parser.add_argument("--dest_path", help="File destination path for model to be saved")
    parser.add_argument("--downsample", type=int, help="Decrease the training data size by sampling every n images")
    parser.add_argument("--eval", type=int, help="Use the saved model")
    parser.add_argument("--unit_test", type=int)
    parser.add_argument("--k", type=int, help="k in knn")
    args = parser.parse_args()

    train_save = "../data/hogTrain.npy"
    test_save = "../data/hogTest.npy"

    if args.use_saved:
        f = open(train_save, "rb")
        train_imgs = np.load(f)
        train_labels = np.load(f)
        f.close() 

        f = open(test_save, "rb")
        test_imgs = np.load(f)
        test_labels = np.load(f)
        f.close()
    else:
        train_imgs, train_labels = dataloader.load_data(args.train_path, preproc="hog")
        test_imgs, test_labels = dataloader.load_data(args.test_path, preproc="hog")
        with open(train_save, "wb") as f:
            np.save(f, train_imgs)
            np.save(f, train_labels)
        with open(test_save, "wb") as f:
            np.save(f, test_imgs)
            np.save(f, test_labels)
        
   
 
    # train_imgs = np.asarray(train_imgs, dtype=np.uint8)
    train_imgs = train_imgs[:, :, 0]
    test_imgs = test_imgs[:, :, 0]

    # test_imgs = np.asarray(test_imgs, dtype=np.uint8)[:, :, 0]

    if args.downsample:
        train_imgs = train_imgs[::args.downsample, :]
        train_labels = train_labels[::args.downsample]
        test_imgs = test_imgs[::args.downsample, :]
        test_labels = test_labels[::args.downsample]
    
    if not args.eval:
        knn = KNeighborsClassifier(n_neighbors=args.k)
        knn.fit(train_imgs, train_labels)
        with open(args.dest_path + "knn.pickle", "wb") as f:
            pickle.dump(knn, f)
    else:
        f = open(args.dest_path + "knn.pickle", "rb")
        knn = pickle.load(f)
        f.close()

    if args.unit_test:
        test_idx = np.random.choice(test_imgs.shape[0], size=args.unit_test)
        test_set = test_imgs[test_idx, :]
        sTime = time.perf_counter()
        pred_label = knn.predict(test_set)
        nTime = time.perf_counter()
        print("Average time taken: ", (nTime - sTime)/args.unit_test)
        test_set_labels = test_labels[test_idx]
        '''for i in range(test_set.shape[0]):
            print("True label: ", test_set_labels[i])
            print("Pred Label: ", pred_label[i])
            currIm = np.reshape(test_set[i, :], (78, 78))
            currIm = exposure.rescale_intensity(currIm, in_range=(0, 10))
            plt.imshow(currIm.T, cmap='gray')
            plt.show()
        im = cv.imread("../data/Test/V/958_V.jpg")
        print(im.shape)
        # im = np.linalg.norm(im, axis=2)
        _, hog_image = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
        # hog = preprocess.extract_HOG(im)
        # print(hog_image)
        plt.imshow(hog_image, cmap='gray')
        plt.show()'''
    else:
        test_pred = knn.predict(test_imgs)

        acc_score = accuracy_score(test_labels, test_pred)
        print("Accuracy: " + str(acc_score))
        print("Confusion matrix: " + str(confusion_matrix(test_labels, test_pred)))




if __name__ == '__main__':
    main()