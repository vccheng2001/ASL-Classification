
''' Decision Tree/Random Forest Classifier '''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


import sys
sys.path.insert(0,'../')
from dataloader import load_data

# X: array-like of shape (n_samples, n_features)
# y: array-like of shape (n_samples,)

print("Loading dataset...")

# load data 
train_path = '../dataset/Train/'
test_path = '../dataset/Test/'
train_x, train_y = load_data(train_path, preproc=None, gray=False)
test_x, test_y = load_data(test_path, preproc=None, gray=False)

print('Train x:', train_x.shape, 'Train y:', train_y.shape)
print('Test x:', test_x.shape, 'Test y:', test_y.shape)

print('Initializing classifier')
clf = RandomForestClassifier(n_estimators=100)

print('Fitting to training data')
clf.fit(train_x, train_y)

print("Predict on test data")
pred_y = clf.predict(test_x)

print("Accuracy: ", accuracy_score(test_y, pred_y))

print('Confusion matrix:')

# create confusion matrix
print(confusion_matrix(test_y, pred_y))
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, test_x, test_y)
print(classification_report(test_y, pred_y))