import trace
import numpy as np
import matplotlib.pyplot as plt
import argparse
import dataloader
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import time


def main():
    parser = argparse.ArgumentParser(description="Compute an SVM model on the dataset.")
    parser.add_argument("--train_path", help="Path to training data")
    parser.add_argument("--test_path", help="Path to test data")
    parser.add_argument("--use_saved", type=int, help="Used the save npy files")
    parser.add_argument("--dest_path", help="File destination path for model to be saved")
    parser.add_argument("--eval", type=int, help="Use the saved model")
    parser.add_argument("--num_in_class", type=int, help="Number of training images in each class")
    parser.add_argument("--unit_test", type=int, help="Number of unit tests to run")
    # parser.add_argument("--dest_name", help="File name for model to be saved as")
    args = parser.parse_args()
    train_save = "../data/dataTrain.npy"
    test_save = "../data/dataTest.npy"
    gtrain_save = "../data/dataTrainGray.npy"
    gtest_save = "../data/dataTestGray.npy"
    if args.use_saved:
        f = open(gtrain_save, "rb")
        train_imgs = np.load(f)
        train_labels = np.load(f)
        f.close() 

        f = open(gtest_save, "rb")
        test_imgs = np.load(f)
        test_labels = np.load(f)
        f.close()
    else:
        train_imgs, train_labels = dataloader.load_data(args.train_path, gray=True)
        test_imgs, test_labels = dataloader.load_data(args.test_path, gray=True)


    print("Loaded data")

    if args.eval == 0:
        # np.random.seed(752)
        # print(train_imgs.shape)
        train_idx = []
        # train_idx = train_idx[::20]

      
        # print(train_idx.shape)

        # Evenly sample from all labels
        count = 0
        prevLabel = 0
        num_counts = 400 # default
        if args.num_in_class is not None:
            num_counts = args.num_in_class

        for i in range(len(train_labels)):
            if prevLabel != train_labels[i]:
                count = 0

            if prevLabel == train_labels[i] and count < num_counts:
                train_idx.append(i)
                count += 1
            
            prevLabel = train_labels[i]


        print(len(train_idx))
            

        sample_train = train_imgs[train_idx, :]
        sample_labels = train_labels[train_idx]
        label_dict = {}
        for label in sample_labels:
            if label not in label_dict:
                label_dict[label] = 0
            label_dict[label] += 1
        
        print(label_dict)
        # print(sample_train.shape)
        # print(np.)

        model = svm.SVC(C=10.0, gamma="scale") # We want plat scaling
        model.fit(sample_train, sample_labels)
        savePath = args.dest_path + "svm.pickle"
        with open(savePath, "wb") as f:
            pickle.dump(model, f)
        print("Training completed")
    else:
        f = open("models/svm.pickle", "rb")
        model = pickle.load(f)
        print("model loaded")
        f.close()

    if not args.unit_test:
    # print(model.best_params_)
        test_pred = model.predict(test_imgs)
        label_dict = {}
        for label in test_labels:
            if label not in label_dict:
                label_dict[label] = 0
            label_dict[label] += 1
        
        print(label_dict)
        acc_score = accuracy_score(test_labels, test_pred)
        print("Accuracy: " + str(acc_score))
        print("Confusion matrix: " + str(confusion_matrix(test_labels, test_pred)))

    else:
        print(test_imgs.shape)
        test_idx = np.random.choice(test_imgs.shape[0], size=args.unit_test)
        test_set = test_imgs[test_idx, :]
        sTime = time.perf_counter()
        pred_label = model.predict(test_set)
        nTime = time.perf_counter()
        print("Average time taken: ", (nTime - sTime) / args.unit_test)
        test_set_labels = test_labels[test_idx]
        for i in range(test_set.shape[0]):
            print("True label: ", test_set_labels[i])
            print("Pred Label: ", pred_label[i])
            plt.imshow(np.reshape(test_set[i, :], (28, 28)))
            plt.show()





    
    
if __name__ == '__main__':
    main()