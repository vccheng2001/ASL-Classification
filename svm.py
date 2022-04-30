import trace
import numpy as np
import matplotlib.pyplot as plt
import argparse
import dataloader
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle



def main():
    parser = argparse.ArgumentParser(description="Compute an SVM model on the dataset.")
    parser.add_argument("--train_path", help="Path to training data")
    parser.add_argument("--test_path", help="Path to test data")
    parser.add_argument("--use_saved", help="Used the save npy files")
    parser.add_argument("--dest_path", help="File destination path for model to be saved")
    parser.add_argument("--eval", type=int, help="Use the saved model")
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

    # g_train_imgs = np.zeros((train_imgs.shape[0], 28*28))
    # g_test_imgs = np.zeros((test_imgs.shape[0], 28*28))

    # for i in range(train_imgs.shape[0]):
    #     t_image = train_imgs[i, :]
    #     t_image = np.reshape(t_image, (28, 28, 3))
    #     # print(np.linalg.norm(t_image, axis=2).shape)
    #     g_train_imgs[i, :] = np.linalg.norm(t_image, axis=2).flatten()
    # for i in range(test_imgs.shape[0]):
    #     t_image = test_imgs[i, :]
    #     t_image = np.reshape(t_image, (28, 28, 3))
    #     g_test_imgs[i, :] =np.linalg.norm(t_image, axis=2).flatten()
        
            
    # with open(gtrain_save, "wb") as f:
    #     np.save(f, g_train_imgs)
    #     np.save(f, train_labels)
    # with open(gtest_save, "wb") as f:
    #     np.save(f, g_test_imgs)
    #     np.save(f, test_labels)
   
    # Decided to save the data instead of reloading it
    #

    print("Loaded data")
    
    # Apply look for the best model using a GridSearchCV search
    # Search over the regularization and use the rbf kernel, I don't think poly helps here

    # # We want to reduce the amount of training data examples, because SVM doesn't require that many in the first place. But we also 
    # label_dict = {}
    # for label in train_labels:
    #     if label not in label_dict:
    #         label_dict[label] = 0
    #     label_dict[label] += 1
    # print(label_dict)
    if args.eval == 0:
        # np.random.seed(752)
        # print(train_imgs.shape)
        train_idx = []
        # train_idx = train_idx[::20]

      
        # print(train_idx.shape)

        # Evenly sample from all labels
        count = 0
        prevLabel = 0
        for i in range(len(train_labels)):
            if prevLabel != train_labels[i]:
                count = 0

            if prevLabel == train_labels[i] and count < 400:
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
        # parameterSearch = {'C':[0.1, 1.0]}
        # model = GridSearchCV(classifier, parameterSearch)
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

    
    
if __name__ == '__main__':
    main()









    